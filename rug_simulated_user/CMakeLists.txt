cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE Debug)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#Find PCL 1.7
find_package(PCL 1.8 REQUIRED)
if (NOT PCL_FOUND)
	MESSAGE("PCL 1.7 not found. You must execute the following command\nsudo apt-get install libpcl-1.7-all-dev\n")
endif (NOT PCL_FOUND)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



#Find CGAL
#<<<<<<< HEAD
# find_package(CGAL REQUIRED)
# include( ${CGAL_USE_FILE} )
# if (NOT CGAL_FOUND)
#     MESSAGE("CGAL not found. Run #sudo apt-get install libcgal*\n")
# endif (NOT CGAL_FOUND)
#=======
find_package(CGAL REQUIRED)
include( ${CGAL_USE_FILE} )
if (NOT CGAL_FOUND)
    MESSAGE("CGAL not found. Run #sudo apt-get install libcgal*\n")
endif (NOT CGAL_FOUND)

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


# # hybrid nodelet library feature extraction
rosbuild_add_library (rug_simulated_user_functionality src/rug_simulated_user_functionality.cpp)
SET_TARGET_PROPERTIES(rug_simulated_user_functionality PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
target_link_libraries(rug_simulated_user_functionality ${PCL_LIBRARIES})
target_link_libraries(rug_simulated_user_functionality perception_db)
target_link_libraries(rug_simulated_user_functionality object_conceptualizer)
target_link_libraries(rug_simulated_user_functionality object_descriptor_functionality)
target_link_libraries(rug_simulated_user_functionality boost_system)


rosbuild_add_executable(rug_simulated_user_hand_crafted_descriptor src/rug_simulated_user_hand_crafted_descriptor.cpp)
SET_TARGET_PROPERTIES(rug_simulated_user_hand_crafted_descriptor PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
target_link_libraries(rug_simulated_user_hand_crafted_descriptor ${PCL_LIBRARIES})
target_link_libraries(rug_simulated_user_hand_crafted_descriptor perception_db)
target_link_libraries(rug_simulated_user_hand_crafted_descriptor boost_system)
target_link_libraries(rug_simulated_user_hand_crafted_descriptor spin_image)
target_link_libraries(rug_simulated_user_hand_crafted_descriptor object_conceptualizer)
target_link_libraries(rug_simulated_user_hand_crafted_descriptor object_descriptor_functionality)
target_link_libraries(rug_simulated_user_hand_crafted_descriptor rug_simulated_user_functionality)

rosbuild_add_executable(rug_simulated_user_deep_learning src/rug_simulated_user_deep_learning.cpp)
SET_TARGET_PROPERTIES(rug_simulated_user_deep_learning PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
target_link_libraries(rug_simulated_user_deep_learning ${PCL_LIBRARIES})
target_link_libraries(rug_simulated_user_deep_learning perception_db)
target_link_libraries(rug_simulated_user_deep_learning boost_system)
target_link_libraries(rug_simulated_user_deep_learning spin_image)
target_link_libraries(rug_simulated_user_deep_learning object_conceptualizer)
target_link_libraries(rug_simulated_user_deep_learning object_descriptor_functionality)
target_link_libraries(rug_simulated_user_deep_learning rug_simulated_user_functionality)



