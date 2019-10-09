// ############################################################################
//    
//   Created: 	1/05/2014
//   Author : 	Hamidreza Kasaei
//   Email  :	seyed.hamidreza@ua.pt
//   Purpose: 	(Purely instance based learning approach)This program follows
//   		the teaching protocol and autonomously interact with the system
//   		using teach, ask and correct actions. For each newly taught category,
//   		the average sucess of the system should be computed. To do that, 
//   		the simulated teacher repeatedly picks object views of the currently
//   		known categories from a database and presents them to the system for 
//   		checking whether the system can recognize them. If not, the simulated
//   		teacher provides corrective feedback.
//   		
// 		This program is part of the RACE project, partially funded by the
//   		European Commission under the 7th Framework Program.
//
//   		See http://www.project-race.eu
// 
//   		(Copyright) University of Aveiro - RACE Consortium
// 
// ############################################################################

/* _________________________________
   |                                 |
   |          RUN SYSTEM BY          |
   |_________________________________| */
   
//rm -rf /tmp/pdb
//roslaunch race_simulated_user simulated_user.launch

/* _________________________________
  |                                 |
  |             INCLUDES            |
  |_________________________________| */
  
//system includes
#include <std_msgs/String.h>
#include <sstream>
#include <fstream>
//ros includes 
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

#include <pluginlib/class_list_macros.h> 
#include <stdio.h>
#include <race_perception_db/perception_db.h>
#include <race_perception_db/perception_db_serializer.h>

//pcl includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ros/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

//package includes
#include <race_perception_msgs/perception_msgs.h>
#include <race_simulated_user/race_simulated_user_functionality.h>
#include <race_perception_utils/print.h>

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <CGAL/Plane_3.h>
#include "std_msgs/Float64.h"

/* _________________________________
  |                                 |
  |            NameSpace            |
  |_________________________________| */

using namespace pcl;
using namespace std;
using namespace ros;
using namespace race_perception_utils;

typedef pcl::PointXYZRGBA PointT;

 /* _________________________________
  |                                 |
  |        Global Parameters        |
  |_________________________________| */

  //dataset
  std::string home_address;	//  IEETA: 	"/home/hamidreza/";
			      //  Washington: "/media/E2480872480847AD/washington/";
 
  int number_of_categories = 49;

/* _________________________________
  |                                 |
  |         Global Variable         |
  |_________________________________| */
  unsigned int cat_id = 1;
  unsigned int track_id =7;
  unsigned int view_id = 1;



int main(int argc, char** argv)
{
    ros::init (argc, argv, "Emulating Object Tracking");
    ros::NodeHandle nh;
    string name = nh.getNamespace();

    /* _________________________________
    |                                 |
    |     Randomly sort categories    |
    |_________________________________| */
    int RunCount=1;
    generateRrandomSequencesCategories(RunCount);
	    
      
    /* ________________________________________
    |                                       |
    |    read prameters from launch file    |
    |_______________________________________| */
    // read database parameter
    nh.param<std::string>("/perception/home_address", home_address, "default_param");
    nh.param<int>("/perception/number_of_categories", number_of_categories, number_of_categories);

    ROS_INFO("home_address = %s", home_address.c_str());
    ROS_INFO("number_of_categories = %d", number_of_categories);

    /* ______________________________
    |                                |
    |         create a publisher     |
    |________________________________| */
    unsigned found = name.find_last_of("/\\");
    std::string pcin_topic = name.substr(0,found) + "/pipeline_default/tracker/tracked_object_point_cloud";  
    ros::Publisher pub = nh.advertise< race_perception_msgs::PCTOV> (pcin_topic, 1000);

    std::string category_name;
    //std::ifstream list_of_categories ("/media/E2480872480847AD/washington/Category/list_of_category.txt");
    
    string path = home_address+ "Category/Category_orginal.txt";
    std::ifstream listOfObjectCategories (path.c_str(), std::ifstream::in);
  //test keypoint selection
    
    while(listOfObjectCategories.good ())
    {
	string categoryAddress;
	std::getline (listOfObjectCategories, categoryAddress);
	if(categoryAddress.empty () || categoryAddress.at (0) == '#') // Skip blank lines or comments
	{
	    continue;
	}
		
	string category_address = home_address +"/"+ categoryAddress.c_str();
	std::ifstream categoryInstancesTmp (category_address.c_str());
	size_t category_size = 0;
	string InstancePath;
	// ROS_INFO("\t\t[-]- TEST0");
	boost::shared_ptr<PointCloud<PointT> > AllPCDFile (new PointCloud<PointT>);
	char ch='1';
	    
	while (categoryInstancesTmp.good ())// read instances of a category 
	{
	    std::getline (categoryInstancesTmp, InstancePath);
	    if(InstancePath.empty () || InstancePath.at (0) == '#') // Skip blank lines or comments
	    {
		continue;
	    }
	    ROS_INFO("\t\t[-] view = %s", InstancePath.c_str());
	    InstancePath = home_address +"/"+ InstancePath.c_str();
	    ROS_INFO("\t\t[-]Instance Path = %s",InstancePath.c_str());
	    std::string ground_truth_category_name =extractCategoryName(InstancePath);
	    ROS_INFO("\t\t[-]Category Name = %s",ground_truth_category_name.c_str());

	    //load an instance from file
	    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
	    if (io::loadPCDFile <PointT> (InstancePath.c_str(), *PCDFile) == -1)
	    {	
		    ROS_ERROR("\t\t[-]-Could not read given object %s :",InstancePath.c_str());
		    return(0);
	    }		   
	    ROS_INFO("\t\t[-]-  track_id: %i , \tview_id: %i ",track_id, view_id );
	    

	    //Declare PCTOV msg 
	    boost::shared_ptr<race_perception_msgs::PCTOV> msg (new race_perception_msgs::PCTOV );
	    pcl::toROSMsg(*PCDFile, msg->point_cloud);
	    msg->track_id = track_id;//it is 
	    msg->view_id = view_id;		    
	    msg->ground_truth_name = ground_truth_category_name;//extractCategoryName(InstancePath);
	    pub.publish (msg);
	    ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", InstancePath.c_str());
	    cin >>ch;

// 	    track_id++;
// 	    view_id ++;
	    category_size++;
	}

	categoryInstancesTmp.close();
	ROS_INFO("\t\t[-]- Category size = %ld", category_size);
	
    }	
	
    
    return 0 ;
}



