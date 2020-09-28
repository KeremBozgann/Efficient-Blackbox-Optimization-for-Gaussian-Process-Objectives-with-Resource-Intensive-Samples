def EI_pu(sigma_x, u_x , f_best, u_cost):
    gama_x= (f_best-u_x)/sigma_x
    fi_x = norm.cdf(gama_x)
    EI_x= sigma_x*(gama_x*fi_x+ norm.pdf(gama_x))

    EI_pu= EI_x/u_cost

    return EI_pu