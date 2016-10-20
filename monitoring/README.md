# Metrics and visualization setup for Clipper

## Overview

Clipper regularly reports metrics associated with system performance and model behavior at consistent time intervals. These metrics are stored in a time series database instance, [InfluxDB](https://www.influxdata.com/time-series-platform/influxdb/). To view and analyze these metrics graphically, we use [Grafana](http://grafana.org/), an open-source visualization library compatible with InfluxDB. The following set of instructions will walk you through the process of installing and running InfluxDB and Grafana in conjunction with Clipper.

## Steps

### Install and start InfluxDB

#### macOS / OS X
1. From a bash prompt, execute the following:

  `$ brew update`

   `$ brew install influxdb`
   
2. Once the InfluxDB installation is complete, start the Influx service by executing

  `$ influxd`
  
3. By default, InfluxDB is hosted locally on port `8086`. Enter InfluxDB's ip address and port number within your Clipper configuration file (`test.toml`, for example) in the corresponding `influx_ip` and `influx_port` fields.

For more information on setting up InfluxDB, see the full [Installation Instructions for OS X](https://docs.influxdata.com/influxdb/v1.0/introduction/installation#mac-os-x).

#### Docker

1. Install the Docker image **tutum/influxdb**. To install via bash, execute

  `$ docker pull tutum/influxdb`
  
2. Proceed to the **Install and start Grafana** section. Complete this before starting Clipper.

### Start Clipper

** If you're using docker, install and start Grafana first (see below) **

1. Start Redis on the port specified in your `.toml` configuration file of choice.
2. Begin serving a model wrapper. For example, execute the following from a bash prompt:

  `$ python clipper/model_wrappers/python/test_new_rpc.py`
  
3. Execute the following command, replacing `conf/test.toml` with the path to your configuration file if necessary.
  `$ RUST_LOG=info RUST_BACKTRACE=1 clipper/clipper_server/target/debug/clipper-rest start --conf conf/test.toml`

### Install and start Grafana

#### macOS / OS X
1. From a bash prompt, execute the following:

  `$ brew install grafana`
  
2. To run Grafana, execute

  `$ brew services start grafana`
  
3. By default, Grafana is hosted locally on port `3000`. Using your browser of choice, visit [http://localhost:3000](http://localhost:3000).

4. Sign up for Grafana by inputting an email address (does not have to be valid when running locally for testing purposes).

5. Click the Grafana logo in the top left corner of the screen, and hover your cursor over the menu item displaying the email address that you signed up with. Select the **New Organization** item (`+` icon).

6. Again, click the Grafana logo and select the **Data Sources** menu item. Click the `Add data source` button. Enter a name for your database, and select "InfluxDB" as the type. Under *Http Settings*, enter the address at which InfluxDB can be accessed ([http://localhost:8086](http://localhost:8086) by default). Under *InfluxDB Details*, enter the name of the database. The database name is the value associated with the `name` field in your `.toml` configuration file; **replace spaces in this name with dashes (for example, "clipper test" becomes "clipper-test")**. Finally, enter a username and password used to authenticate with InfluxDB; unless you have set up specific credentials for authorization, entering any username and password should work. Click `Save & Test`.

7. Click the Grafana logo and hover your cursor over the **Dashboards** menu item. Select the **Import** item (`+` icon). Click the "Upload .json File" button and select the dashboard JSON template located at `clipper/monitoring/grafana_dashboard.json`. Enter a name for your dashboard and select the InfluxDB data source you added in step 6. Finally, click the `Save & Open` button.

8. You should now see a dashboard that includes visualizations for simple metrics like predictions over time, throughput, and latency.

For more information on getting started with Grafana, see the full [Getting Started Guide](http://docs.grafana.org/guides/gettingstarted/).

#### Docker

1. Install the Grafana Docker image **tutum/influxdb**. To install via bash, execute

  `$ docker pull grafana/grafana`
 
2. Execute the **init_metrics.sh** script via bash to start and set up InfluxDB and Grafana:
  
  `$ ./clipper/monitoring/init_metrics.sh
  
3. Navigate to ([http://localhost:3000](http://localhost:3000) and log in with username "admin" and the password "admin".

4. On Grafana home screen, you should see a "Clipper Metrics" link in the list of recently viewed dashboards. Click the link to access the default metrics dashboard.

5. Start Clipper (see section above)

### 
