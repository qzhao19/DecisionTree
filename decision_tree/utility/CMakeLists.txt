target_sources(dtreelib 
    INTERFACE FILE_SET HEADERS 
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES math.hpp random.hpp sort.hpp)