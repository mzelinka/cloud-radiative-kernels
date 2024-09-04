# cloud-radiative-kernels

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13686878.svg)](https://doi.org/10.5281/zenodo.13686878)

Cloud radiative kernels are provided, along with a Jupyter notebook that demonstrates how to use them to compute cloud feedbacks.  The notebook additionally demonstrates how to break the feedback down into cloud amount, altitude, optical depth, and residual components for all clouds, non-low clouds, and low clouds. The notebook also demonstrates how to account for changing obscuration effects.

References
----------
Zelinka, M. D., S. A. Klein, and D. L. Hartmann, 2012: Computing and Partitioning Cloud Feedbacks Using 
    Cloud Property Histograms. Part I: Cloud Radiative Kernels. J. Climate, 25, 3715-3735. 
    [doi:10.1175/JCLI-D-11-00248.1](http://journals.ametsoc.org/doi/abs/10.1175/JCLI-D-11-00248.1).

Zelinka, M. D., S. A. Klein, and D. L. Hartmann, 2012: Computing and Partitioning Cloud Feedbacks Using 
    Cloud Property Histograms. Part II: Attribution to Changes in Cloud Amount, Altitude, and Optical Depth. 
    J. Climate, 25, 3736-3754. [doi:10.1175/JCLI-D-11-00249.1](http://journals.ametsoc.org/doi/abs/10.1175/JCLI-D-11-00249.1).

Zelinka, M.D., S.A. Klein, K.E. Taylor, T. Andrews, M.J. Webb, J.M. Gregory, and P.M. Forster, 2013: 
    Contributions of Different Cloud Types to Feedbacks and Rapid Adjustments in CMIP5. 
    J. Climate, 26, 5007-5027. [doi:10.1175/JCLI-D-12-00555.1](http://journals.ametsoc.org/doi/abs/10.1175/JCLI-D-12-00555.1).
    
Zelinka, M. D., C. Zhou, and S. A. Klein, 2016: Insights from a Refined Decomposition of Cloud Feedbacks, 
    Geophys. Res. Lett., 43, 9259-9269, [doi:10.1002/2016GL069917](http://onlinelibrary.wiley.com/doi/10.1002/2016GL069917/full).
    
Zhou, C., M. D. Zelinka, A. E. Dessler, P. Yang, 2013: An analysis of the short-term cloud feedback using 
    MODIS data, J. Climate, 26, 4803–4815. [doi:10.1175/JCLI-D-12-00547.1](http://journals.ametsoc.org/doi/abs/10.1175/JCLI-D-12-00547.1).


Input
----------

The code makes use of the following data:

| Frequency | Name | Description | Unit | File Format |
|:----------|:-----------------------------|:-------------|:------|:------------|
| monthly mean | clisccp | ISCCP simulator cloud fraction histograms | % | nc |
| monthly mean | rsuscs | upwelling SW flux at the surface under clear skies | W/m^2 | nc |
| monthly mean | rsdscs | downwelling SW flux at the surface under clear skies | W/m^2 | nc |
| monthly mean | tas | surface air temperature | K | nc |
| monthly mean | LWkernel | LW cloud radiative kernel | W/m^2/% | nc |
| monthly mean | SWkernel | SW cloud radiative kernel | W/m^2/% | nc |

Two sets of cloud radiative kernels available at https://github.com/mzelinka/cloud-radiative-kernels/tree/master/data

1) cloud_kernels2.nc: The cloud radiative kernels developed using zonal mean temperature and humidity profiles averaged across control runs of six CFMIP1 climate models as input to the radiation code. These are best for diagnosing feedbacks relative to a modeled pre-industrial climate state. Please refer to Zelinka et al. (2012a,b) for details.

2) obs_cloud_kernels3.nc: The cloud radiative kernels developed using zonal mean temperature, humidity, and ozone profiles from ERA Interim over the period 2000-2010 as input to the radiation code. These are best for diagnosing feedbacks relative to an observed present-day climate state. Please refer to Zhou et al. (2013) for details.
