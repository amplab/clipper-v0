# Parse configuration files for metrics database information

name_field="influx_dbname"
ip_field="influx_ip"
port_field="influx_port"
regex='(\".+\")'

while read p; do
  if [[ $p == *$name_field* ]]
  then
    [[ $p =~ $regex ]] 
    parsed=${BASH_REMATCH[1]}
    dbname=${parsed:1:${#parsed} - 2}
  elif [[ $p == *$ip_field* ]]
  then
    [[ $p =~ $regex ]]
    parsed=${BASH_REMATCH[1]}
    influxip=${parsed:1:${#parsed} - 2}	
  elif [[ $p == *$port_field* ]]
  then
    [[ $p =~ ([0-9]+) ]]
    influxport=${BASH_REMATCH[1]}
  fi
done < "../clipper_server/conf/test.toml"

# Start InfluxDB
echo -e "Starting InfluxDB ..."
docker run -d -p 8083:8083 -p $influxport:$influxport tutum/influxdb > /dev/null
sleep 5

# Start Grafana
echo -e "Starting Grafana ..."
docker run -d -p 3000:3000 grafana/grafana > /dev/null
sleep 10

# Configure Grafana
influx_url="http://$influxip:$influxport"
echo -e "Adding metrics data source for database $dbname running at $influx_url ..."

curl -s 'http://admin:admin@127.0.0.1:3000/api/datasources' -X POST -H 'Content-Type: application/json;charset=UTF-8' --data-binary '{"name":"Clipper Metrics","type":"influxdb","url":"'"$influx_url"'","access":"direct","isDefault":true,"database":"'"$dbname"'","user":"admin","password":"admin"}' > /dev/null

echo -e "Creating default metrics dashboard for database $dbname running at $influx_url ..."

dashboard_json=`cat grafana_dashboard.json`
dashboard_request='{"dashboard":'$dashboard_json'}'
rm dashreq.json &> /dev/null
echo $dashboard_request > "dashreq.json"

curl -s 'http://admin:admin@127.0.0.1:3000/api/dashboards/db' -X POST -H 'Content-Type: application/json;charset=UTF-8' -d @dashreq.json > /dev/null

rm dashreq.json
