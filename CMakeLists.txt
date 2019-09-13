cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE Release)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#Find PCL 1.8
find_package(PCL 1.8 REQUIRED)
if (NOT PCL_FOUND)
	MESSAGE("PCL 1.8 not found.")
endif (NOT PCL_FOUND)

#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})

### If you use ESF shape descriptor, comment the following line
add_definitions(${PCL_DEFINITIONS})


#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

# hybrid nodelet library feature extraction
rosbuild_add_library(rug_kfold_cross_validation_functionality src/rug_kfold_cross_validation_functionality.cpp)
SET_TARGET_PROPERTIES(rug_kfold_cross_validation_functionality PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
target_link_libraries(rug_kfold_cross_validation_functionality ${PCL_LIBRARIES})
target_link_libraries(rug_kfold_cross_validation_functionality object_descriptor_functionality)
#target_link_libraries(rug_kfold_cross_validation_functionality object_conceptualizer)
target_link_libraries(rug_kfold_cross_validation_functionality boost_system)


rosbuild_add_executable(k_fold_cross_validation src/k_fold_cross_validation.cpp)
SET_TARGET_PROPERTIES(k_fold_cross_validation PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
target_link_libraries(k_fold_cross_validation ${PCL_LIBRARIES} ${PCL_INCLUDE_DIRS})
target_link_libraries(k_fold_cross_validation object_descriptor_functionality)
target_link_libraries(k_fold_cross_validation rug_kfold_cross_validation_functionality)
target_link_libraries(k_fold_cross_validation perception_db)
target_link_libraries(k_fold_cross_validation boost_system)
target_link_libraries(k_fold_cross_validation spin_image)
target_link_libraries(k_fold_cross_validation object_conceptualizer)


## MODELNET
rosbuild_add_executable(k_fold_cross_validation_deep_learning_descriptor src/k_fold_cross_validation_deep_learning_descriptor.cpp)
SET_TARGET_PROPERTIES(k_fold_cross_validation_deep_learning_descriptor PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
target_link_libraries(k_fold_cross_validation_deep_learning_descriptor ${PCL_LIBRARIES} ${PCL_INCLUDE_DIRS})
target_link_libraries(k_fold_cross_validation_deep_learning_descriptor object_descriptor_functionality)
target_link_libraries(k_fold_cross_validation_deep_learning_descriptor rug_kfold_cross_validation_functionality)
target_link_libraries(k_fold_cross_validation_deep_learning_descriptor perception_db)
target_link_libraries(k_fold_cross_validation_deep_learning_descriptor boost_system)
target_link_libraries(k_fold_cross_validation_deep_learning_descriptor spin_image)
target_link_libraries(k_fold_cross_validation_deep_learning_descriptor object_conceptualizer)





