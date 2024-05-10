### Use case


    import geopandas as gpd
    from generate_od import generator
    
    if __name__ == "__main__":
    
        my_generator = generator.Generator()
        my_generator.set_satetoken("xxxxxxxxxxxxxxx")
    
        area = gpd.read_file("London.shp")
        my_generator.load_area(area)
        my_generator.generate()

