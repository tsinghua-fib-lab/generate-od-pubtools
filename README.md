# Generate Origin-destination Matrix based-on Public Available Information on the Internet
The following code can be used to automatically obtain Population County (from WorldPop [https://hub-worldpop.opendata.arcgis.com/]) and satellite imagery (World_Imagery [https://www.arcgis.com/home/item.html?id=10df2279f9684e4a9f6a7f08febac2a9]) for a given area divided by several regions. Based on this information, it can generate the origin-destination (OD) matrix between regions using a graph denoising diffusion model.

### Use case

    import geopandas as gpd
    from generate_od import generator
    
    if __name__ == "__main__":
    
        my_generator = generator.Generator()
        my_generator.set_satetoken("xxxxxxxxxxxxxxx")
    
        area = gpd.read_file("London.shp")
        my_generator.load_area(area)
        my_generator.generate()

