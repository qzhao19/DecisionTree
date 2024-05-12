target_sources(dtreelib 
    INTERFACE FILE_SET HEADERS 
    BASE_DIRS ${PROJECT_SOURCE_DIR}
    FILES builder.hpp splitter.hpp tree.hpp)