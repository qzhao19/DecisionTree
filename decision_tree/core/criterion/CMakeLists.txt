target_sources(dtreelib 
    INTERFACE FILE_SET HEADERS 
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES base.hpp entropy.hpp gini.hpp)