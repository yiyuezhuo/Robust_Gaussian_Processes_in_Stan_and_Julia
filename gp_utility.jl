
using Plots
using ColorSchemes
using CmdStanExtract
using Statistics

function scatter_total!(x_total, y_total)
    scatter!(x_total, y_total, markersize=2., markercolor=:green, markerstrokecolor=:white)
end

function scatter_obs!(x, y)
    scatter!(x, y, markersize=4., markercolor=:black, markerstrokecolor=:white)
end

function plot_main!(x,y)
    plot!(x, y, width=2, color=:black)
end

function plot_base!(data, true_realization)
    scatter_total!(data.x_predict, data.y_predict)
    plot_main!(true_realization.x_total, true_realization.f_total)
    scatter_obs!(data.x, data.y)
end

function plot_between!(x, y1, y2; fillalpha=nothing, fillcolor=:match)
    y = (y1+y2) / 2
    plot!(x, y, ribbon=(y - y1, y2 - y), linewidth=0; fillalpha, fillcolor)
end

function plot_gp_pred_quantiles(fit, data, true_realization, title)
    
    params = extract(fit)
    
    f_total = true_realization.f_total
    x_total = true_realization.x_total
    
    x_predict = data.x_predict
    # y_predict = data.y_predict
    
    q_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_list = hcat([quantile(params["y_predict"][i,:,1], q_vec) for i in 1:size(params["y_predict"], 1)]...)
    
    plot(legend=false)
    
    plot_between!(x_predict, q_list[1,:], q_list[end,:], fillcolor=colorschemes[:amp][75])
    plot_between!(x_predict, q_list[2,:], q_list[end-1,:], fillcolor=colorschemes[:amp][100])
    plot_between!(x_predict, q_list[3,:], q_list[end-2,:], fillcolor=colorschemes[:amp][125])
    plot_between!(x_predict, q_list[4,:], q_list[end-3,:], fillcolor=colorschemes[:amp][150])
    plot!(x_predict, q_list[5,:], linewidth=2, color=colorschemes[:amp][175])
    
    plot_base!(data, true_realization)
    
    title!(title)
end

function plot_gp_quantiles(fit, data, true_realization, title)
    
    params = extract(fit)
    
    f_total = true_realization.f_total
    x_total = true_realization.x_total
    
    x_predict = data.x_predict
    # y_predict = data.y_predict
    
    q_vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_list = hcat([quantile(params["f_predict"][i,:,1], q_vec) for i in 1:size(params["f_predict"], 1)]...)
    
    plot(legend=false)
    
    plot_between!(x_predict, q_list[1,:], q_list[end,:], fillcolor=colorschemes[:amp][75])
    plot_between!(x_predict, q_list[2,:], q_list[end-1,:], fillcolor=colorschemes[:amp][100])
    plot_between!(x_predict, q_list[3,:], q_list[end-2,:], fillcolor=colorschemes[:amp][125])
    plot_between!(x_predict, q_list[4,:], q_list[end-3,:], fillcolor=colorschemes[:amp][150])
    plot!(x_predict, q_list[5,:], linewidth=2, color=colorschemes[:amp][175])
    
    plot_base!(data, true_realization)
    
    title!(title)
end

function plot_gp_realizations(fit, data, true_realization, title)
    params = extract(fit)
    
    plot(fmt=:png, legend=false)
    
    plot!(data.x_predict, params["f_predict"][:,:,1], color=colorschemes[:amp][150], alpha=0.05)
    plot_base!(data, true_realization)
    
    # plot_base!(data, true_realization)
    
    title!(title)
end

function plot_gp_pred_realizations(fit, data, true_realization, title)
    params = extract(fit)
    
    plot(fmt=:png, legend=false)
    
    plot!(data.x_predict, params["y_predict"][:,:,1], color=colorschemes[:amp][150], alpha=0.05)
    plot_base!(data, true_realization)
    
    # plot_base!(data, true_realization)
    
    title!(title)
end