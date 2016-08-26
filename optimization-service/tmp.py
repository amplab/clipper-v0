def process_batch(name, batches, latencies):
#     quantiles = np.arange(0.09, 0.999, 0.1)
    quantiles = [0.9, 0.95, 0.99]
    
    
    df = pd.DataFrame({"batch_size": batches, "latencies": latencies})
#     print df.head()
    mod = smf.quantreg('latencies ~ batch_size', df)
    def fit_model(q):
        model_fit = mod.fit(q=q)
        return [q, model_fit.params['Intercept'], model_fit.params['batch_size']] + model_fit.conf_int().ix['batch_size'].tolist()    
    models = [fit_model(cur_q) for cur_q in quantiles]
    models = pd.DataFrame(models, columns=['q', 'a', 'b','lb','ub'])
    
    ols = smf.ols('latencies ~ batch_size', df).fit()
    ols_ci = ols.conf_int().ix['batch_size'].tolist()
    ols = dict(a = ols.params['Intercept'],
               b = ols.params['batch_size'],
               lb = ols_ci[0],
               ub = ols_ci[1])
    
#     print models
#     print ols

    print results[model]["name"], results[model]["batch_size"]
#     fig, ax = plt.subplots(figsize=(8,6))
    
    
    x_max_lim = 450
    x = np.arange(0, x_max_lim, 1)
    get_y = lambda a, b: a + b * x

    fig, ax = plt.subplots(figsize=(8, 6))
    lw=2
    ax.scatter(batches, latencies, alpha=0.3)
    for i in range(models.shape[0]):
        y = get_y(models.a[i], models.b[i])
        ax.plot(x, y, linestyle='dotted', color='black', linewidth=lw, label="p%d" % int(models.q[i]*100))
#         ax.plot(x, y, color='black')


    y = get_y(ols['a'], ols['b'])

    ax.plot(x, y, color='red', label='OLS', linewidth=lw,)
    predicted_batch_size = results[model]["batch_size"]
#     ax.scatter(data.income, data.foodexp, alpha=.2)
    ax.plot(np.ones(2)*predicted_batch_size, [0, 100000], color="blue", linewidth=lw, linestyle="dotted", label= "predicted batch size")
    ax.plot([0, x_max_lim], np.ones(2)*50000, color="green", linestyle="dotted", linewidth=lw, label = "latency objective")

    ax.set_xlim((0, x_max_lim))
    ax.set_ylim((0, 100000))
    
    
    
    
    
    legend = ax.legend(loc=0)
#     ax.set_xlabel('Income', fontsize=16)
#     ax.set_ylabel('Food expenditure', fontsize=16);
    ax.set_title(name)
    ax.set_ylabel("latency")
    ax.set_xlabel("batch size")
    plt.show()



for model in results:
#     print results[model].keys()
    batches = []
    latencies = []
    for m in results[model]["measurements"]:
        latencies.append(m["latency"])
        batches.append(m["batch_size"])
    process_batch(model, batches, latencies)