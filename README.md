# Code for net migration & exposure analyses

## Input data for the scripts
* Net migration admin-0 and admin-1 geospatial files from [Niva et al. (2023)](http://doi.org/10.5281/zenodo.7997134)
* Population data from [WorldPop](https://hub.worldpop.org/geodata/listing?id=64)
* ISIMIP 3a model output data from [Zantout et al. (2025)](https://www.nature.com/articles/s41467-025-65600-7)

## Code structure

The code is divided into the following sections:

*  **01_build_admin_boundaries**: Generates a mixed admin layer: countries with mean ADM-1 area < 15,000 km² and countries without ADM-1 are collapsed to ADM-0; all others remain at ADM-1. This standardizes spatial units for consistent comparisons.

*  **02_aggregate_migration**: Aggregates net migration data to mixed admin boundaries

*  **03_aggregate_worldpop**: Aggregates (fractional area weight) WorldPop 1 km rasters to mixed admin boundaries

*  **04_migration_counts**: Weights net migration rates data with WorldPop data (multiply them) to get population counts

*  **05_aggregate_heatwaves**: Aggregates (fractional area weight) heatwave hazard exposure 5 arc-min resolution data to mixed admin boundaries

*  **06_aggregate_crop_failure**: Same as above, for crop failure 

*  **07_aggregate_wildfires**: Same as above, for wildfires

*  **08_map_migration**: Creates map of average annual net migration rates on mixed admin boundaries

*  **09_map_hazards**: Creates maps of average WorldPop population

*  **10_map_hazards**: Creates maps of average annual population exposure to hazards

*  **11_calculate_land_area_heatwaves**: Calculates global average land area exposed to heatwaves

*  **12_map_regional_patterns**: Creates four maps per hazard showing population exposure × net migration, separated by in/out-migration and increasing/decreasing exposure

*  **13_compare models**: Compares multiple models by population exposed to hazards in terms of migration counts



