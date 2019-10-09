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
#include <feature_extraction/spin_image.h>
#include <race_perception_msgs/perception_msgs.h>
#include <object_conceptualizer/object_conceptualization.h>
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
  |        Global Parameters       |
 |_________________________________| */

    //dataset
    std::string home_address;	//  IEETA: 	"/home/hamidreza/";
				//  Washington: "/media/E2480872480847AD/washington/";
    //spin images parameters
    int    spin_image_width = 8 ;
    double spin_image_support_lenght = 0.2;
    int    subsample_spinimages = 10;

    //simulated user parameters
    double P_Threshold = 0.67;  
    int user_sees_no_improvment_const = 100;
    int window_size = 3;
    int number_of_categories =49 ;

    std::string name_of_approach = "VFH";

/* _________________________________
  |                                 |
  |         Global Variable         |
  |_________________________________| */

PerceptionDB* _pdb;
typedef pcl::PointXYZRGBA PointT;

unsigned int cat_id = 1;
unsigned int track_id =1;
unsigned int view_id = 1;
string InstancePathTmp= "";

std::string PCDFileAddressTmp;
std::string True_Category_Global;
std::string Object_name_orginal;
std::string evaluationFile, evaluationTable, precision_file;

std::ofstream Result , Result_table , PrecisionMonitor, PrecisionEpoch, NumberofFeedback , category_random, instances_random, category_introduced;
int TP =0, FP=0, FN=0, TPtmp =0, FPtmp=0, FNtmp=0, Obj_Num=0, number_of_instances=0;//track_id_gloabal = 1 ;//, track_id_gloabal2=1;      

float PrecisionSystem =0;
vector <int> recognition_results; // we coded 0: continue(correctly detect unkown object)
				  // 1: TP , 2: FP 
				  //3: FN , 4: FP and FN
				  


void evaluationfunction(const race_perception_msgs::RRTOV &result)
{
    PrettyPrint pp;
    string tmp = result.ground_truth_name.substr(home_address.size(),result.ground_truth_name.size());
    InstancePathTmp = tmp;

    string True_cat = extractCategoryName(tmp);
    
    std:: string Object_name;
    Object_name = extractObjectName (result.ground_truth_name);

//     Object_name = extractObjectName (Object_name_orginal);
    pp.info(std::ostringstream().flush() << "extractObjectName: "<< Object_name.c_str()); 
    
    Obj_Num++;
    pp.info(std::ostringstream().flush() << "track_id="<<result.track_id << "\tview_id=" << result.view_id);
    
    pp.info(std::ostringstream().flush() << "normalizedObjectCategoriesDistance{");
    for (size_t i = 0; i < result.result.size(); i++)
    {
	pp.info(std::ostringstream().flush() << "-"<< result.result.at(i).normalized_distance);
    }
    pp.info(std::ostringstream().flush() << "}\n");

    int categoryIndex =-1;  
    
    if ( result.result.size() <= 0)
    {
	pp.warn(std::ostringstream().flush() << "Warning: Size of NormalObjectToCategoriesDistances is 0");
    }
       
    float minimumDistance = result.minimum_distance;
 
    string Predict_cat;
    Predict_cat= result.recognition_result.c_str();
   
    pp.info(std::ostringstream().flush() << "[-]Object_name: "<< Object_name.c_str());
    pp.info(std::ostringstream().flush() << "[-]True_cat: "<<True_cat.c_str());
    pp.info(std::ostringstream().flush() << "[-]Predict_cat: " << Predict_cat.c_str());

    char Unknown[] = "Unknown";
   
    Result.open( evaluationFile.c_str(), std::ofstream::app);    
    Result.precision(4);
   
    if ((strcmp(True_cat.c_str(),Unknown)==0) && (strcmp(Predict_cat.c_str(),Unknown)==0))
    { 	
	recognition_results.push_back(0);// continue
	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t0\t0"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
    }
    else if ((strcmp(True_cat.c_str(),Predict_cat.c_str())==0))
    { 
	TP++;
	TPtmp++;
	recognition_results.push_back(1);
	
	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "1\t0\t0" << "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
    }
    else if ((strcmp(True_cat.c_str(),Unknown)==0) && (strcmp(Predict_cat.c_str(),Unknown)!=0))
    { 	
	FP++; 
	FPtmp++;
	recognition_results.push_back(2);

	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t1\t0"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
    }
    else if ((strcmp(True_cat.c_str(),Unknown)!=0) && (strcmp(Predict_cat.c_str(),Unknown)==0))
    { 	
	FN++;
	FNtmp++;
	recognition_results.push_back(3);

	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t0\t1"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
	Result << "\n\n\t.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
	Result << "\n\t - Correct classifier - Category Updated";
	Result << "\n\t .<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.\n\n";
	number_of_instances ++;

	IntroduceNewInstanceVFH (InstancePathTmp, 
				cat_id, 
				track_id, 
				view_id);
	track_id++;
	pp.info(std::ostringstream().flush() << "[-]Category Updated");
    }
    else if ((strcmp(True_cat.c_str(),Predict_cat.c_str())!=0))
    {  	
	FP++; FN++;
	FPtmp++; FNtmp++;    
	recognition_results.push_back(4);

	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t1\t1"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
	Result << "\n\n\t.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
	Result << "\n\t - Correct classifier - Category Updated";
	Result << "\n\t .<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.\n\n";	
	number_of_instances++;
	IntroduceNewInstanceVFH (InstancePathTmp, 
				 cat_id, 
				 track_id, 
				 view_id);
	track_id++;
	pp.info(std::ostringstream().flush() << "[-]Category Updated");
    }
    Result.close();
    Result.clear();
    pp.printCallback();
}



int main(int argc, char** argv)
{
    int RunCount=1;
    
//     while(RunCount <= number_of_categories)
//     {
	/* __________________________________
	|                                   |
	|  Creating a folder for each RUN  |
	|_________________________________| */
	//int to string converting 
	char run_count [10];
	sprintf( run_count, "%d",RunCount );
	//Creating a folder for each RUN.
	string systemStringCommand= "mkdir "+ ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
	system( systemStringCommand.c_str());
	
	/* _________________________________
	|                                  |
	|     Randomly sort categories    |
	|_________________________________| */
 	generateRrandomSequencesCategories(RunCount);
	
	evaluationFile = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count + "/Detail_Evaluation.txt";
	Result.open (evaluationFile.c_str(), std::ofstream::out);
	//Result <<"\nclassification_threshold = " << classification_threshold <<  "\nspin_image_width = " << spin_image_width << "\nsubsample_spinimages = " << subsample_spinimages << "\n\n";
	Result << "\nNum"<<"\tObject_name" <<"\t\t\t"<< "True_Category" <<"\t\t"<< "Predict_Category"<< "\t"<< "TP" << "\t"<< "FP"<< "\t"<< "FN \t\tDistance";
	Result << "\n------------------------------------------------------------------------------------------------------------------------------------";
	Result.close();
	Result.clear();
	
	precision_file = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count +"/PrecisionMonitor.txt";
	PrecisionMonitor.open (precision_file.c_str(), std::ofstream::trunc);
	PrecisionMonitor.precision(4);
	PrecisionMonitor.close();
		
	string path_tmp = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count + "/Category_Introduced.txt";
	category_introduced.open (path_tmp.c_str(), std::ofstream::out);
	
	ros::init (argc, argv, "EVALUATION");
	
	ros::NodeHandle nh;
	_pdb = race_perception_db::PerceptionDB::getPerceptionDB(&nh); //initialize the database class_list_macros
	string name = nh.getNamespace();
	
	/* ________________________________________
	  |                                       |
	 |     read prameters from launch file   |
	|_______________________________________| */
	// read database parameter
	nh.param<std::string>("/perception/home_address", home_address, "default_param");
	nh.param<int>("/perception/number_of_categories", number_of_categories, number_of_categories);

	//read spin images parameters
	nh.param<int>("/perception/spin_image_width", spin_image_width, spin_image_width);
	nh.param<double>("/perception/spin_image_support_lenght", spin_image_support_lenght, spin_image_support_lenght);
	nh.param<int>("/perception/subsample_spinimages", subsample_spinimages, subsample_spinimages);
	
	//read simulated teacher parameters
	nh.param<double>("/perception/P_Threshold", P_Threshold, P_Threshold);
	nh.param<int>("/perception/user_sees_no_improvment_const", user_sees_no_improvment_const, user_sees_no_improvment_const);
	nh.param<int>("/perception/window_size", window_size, window_size);	
	
	
	//start tic	
	ros::Time beginProc = ros::Time::now(); 
	/* _________________________________
	  |                                |
	 |       wait for 0.5 second      |
	|________________________________| */
	
	ros::Time start_time = ros::Time::now();
	while (ros::ok() && (ros::Time::now() - start_time).toSec() < 0.5)
	{  //wait  
	}
		    
	/* ____________________________________________________
	  |                                                   |
	 |   create a subscriber to get recognition feedback |
	|___________________________________________________| */
	
	unsigned found = name.find_last_of("/\\");
	std::string topic_name = name.substr(0,found) + "/tracking/recognition_result";
	ros::Subscriber sub = nh.subscribe(topic_name, 10, evaluationfunction);

	/* _________________________________
	|                                  |
	|         create a publisher      |
	|________________________________| */
	// 
	std::string pcin_topic = name.substr(0,found) + "/pipeline_default/tracker/tracked_object_point_cloud";  
	ros::Publisher pub = nh.advertise< race_perception_msgs::PCTOV> (pcin_topic, 1000);

	
	/* _________________________________
	  |                                |
	 |         Initialization         |
	|________________________________| */
	
	string categoryName="";
	string InstancePath= "";	
	int class_index = 1; // class index
	unsigned int instance_number = 1;
	cat_id = 1;// it is a constant to create key for categories <Cat_Name><Cat_ID>
	track_id = 1;
	view_id = 1; // it is a constant to create key for categories <TID><VID> (since the database samples were
		    // collected manually, each track_id has exactly one view)

	vector <unsigned int> instance_number2;
	for (int i = 0; i < number_of_categories ; i++)
	{
	    instance_number2.push_back(1);
	}
	    
	int number_of_taught_categories = 0;

	/* _________________________________
	  |                                |
	 |        Introduce category      |
	|________________________________| */

	introduceNewCategoryVFH(class_index,
				track_id,
				instance_number2.at(class_index-1),
				evaluationFile
				);
	
	number_of_instances+=3;
	number_of_taught_categories ++;
	category_introduced<< "1\n";

	/* _________________________________
	  |                                |
	 |        Simulated Teacher       |
	|________________________________| */
	
	categoryName = "";    
	float Precision = 0;
	vector <float> average_class_precision;
	
	while ( class_index < number_of_categories)  // one category already taught above
	{
	    class_index ++; // class index
	    InstancePath= "";
	    if (introduceNewCategoryVFH( class_index,
					 track_id,
					 instance_number2.at(class_index-1),
					 evaluationFile) == -1)
	    {
		
		ros::Duration duration = ros::Time::now() - beginProc;
		report_current_results(TP,FP,FN,evaluationFile,true);
		report_experiment_result ( average_class_precision,
					    number_of_instances, 
					    number_of_taught_categories,  
					    evaluationFile, duration);
		
		category_introduced.close();	
		report_all_experiments_results (TP, FP, FN, Obj_Num,			    
						average_class_precision,
						number_of_instances, 
						number_of_taught_categories,
						name_of_approach );
		
		Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);
		systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
		system( systemStringCommand.c_str());

		return (0) ;
	    }
		
	    number_of_instances+=3;
	    number_of_taught_categories ++;
	    category_introduced<< "1\n";
	    categoryName = "";	    
	    TPtmp = 0; FPtmp = 0; FNtmp = 0;
	    
	    int k = 0; // number of classification results
	    float Precision_tmp = 0;
	    Precision = 0;
	    unsigned int c =1;// class index
	    int iterations =1;
	    int iterations_user_sees_no_improvment =0;
	    bool User_sees_no_improvement_in_precision = false;	// In the current implementation, If the simulated teacher 
								// sees the precision doesn't improve in 100 iteration, then, 
								// it terminares evaluation of the system, originally, 
								// it was an empirical decision of the human instructor
	    
	    while ( ((Precision < P_Threshold ) or (k < number_of_taught_categories))
		    and (!User_sees_no_improvement_in_precision) )
	    {
		category_introduced<< "0\n";
		ROS_INFO("\t\t[-] Iteration:%i",iterations);
		ROS_INFO("\t\t[-] c:%i",c);
		ROS_INFO("\t\t[-] Instance number:%i",instance_number2.at(c-1));
		//info for debug
		ROS_INFO("\t\t[-] Home address parameter : %s", home_address.c_str());
		ROS_INFO("\t\t[-] number_of_categories : %i", number_of_categories);
		ROS_INFO("\t\t[-] spin_image_width : %i", spin_image_width);
		ROS_INFO("\t\t[-] spin_image_support_lenght : %lf", spin_image_support_lenght);
		ROS_INFO("\t\t[-] subsample_spinimages : %i", subsample_spinimages);
		ROS_INFO("\t\t[-] P_Threshold : %lf", P_Threshold);
		ROS_INFO("\t\t[-] user_sees_no_improvment_const : %i", user_sees_no_improvment_const);
		ROS_INFO("\t\t[-] window_size : %i", window_size);

		// select an instance from an specific category
		InstancePath= "";
		selectAnInstancefromSpecificCategory(c, instance_number2.at(c-1), InstancePath);
		ROS_INFO("\t\t[-]-Test Instance: %s", InstancePath.c_str());
    
		// check the selected instance exist or not? if yes, send it to the race_feature_extractor
		if (InstancePath.size() < 2) 
		{
		    ROS_INFO("\t\t[-]-The %s file does not exist", InstancePath.c_str());
		    category_introduced.close();
		    ROS_INFO("\t\t[-]- number of taught categories= %i", number_of_taught_categories); 
		    
		    ros::Duration duration = ros::Time::now() - beginProc;
		    report_current_results(TP,FP,FN,evaluationFile,true);
		   
		    report_experiment_result (average_class_precision,
					      number_of_instances, 
					      number_of_taught_categories,  
					      evaluationFile, duration);		    		    
    	
		    report_all_experiments_results (TP, FP, FN, Obj_Num,			    
				    average_class_precision,
				    number_of_instances, 
				    number_of_taught_categories,
				    name_of_approach );
		    
		    Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);

		    systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
		    system( systemStringCommand.c_str());
		    

		    
		    return (0) ;	    
		}
		else
		{
		    std::string ground_truth_category_name =extractCategoryName(InstancePath);
		    InstancePath = home_address +"/"+ InstancePath.c_str();
		    
		    //load an instance from file
		    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
		    if (io::loadPCDFile <PointXYZRGBA> (InstancePath.c_str(), *PCDFile) == -1)
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
		    msg->ground_truth_name = InstancePath;//extractCategoryName(InstancePath);
		    pub.publish (msg);
		    ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", InstancePath.c_str());
		
		    //test keypoint selection
// 		    char ch='1';
// 		    boost::shared_ptr<race_perception_msgs::PCTOV> msg (new race_perception_msgs::PCTOV );
// 		    while (ch !='0')
// 		    {   
// 			pcl::toROSMsg(*PCDFile, msg->point_cloud);
// 			msg->track_id = track_id;//it is 
// 			msg->view_id = view_id;		    
// 			msg->ground_truth_name = InstancePath;//extractCategoryName(InstancePath);
// 			pub.publish (msg);
// 			ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", InstancePath.c_str());
// 			cin >>ch;
// 		    }
		    
		    start_time = ros::Time::now();
		    while (ros::ok() && (ros::Time::now() - start_time).toSec() < 1)
		    { //wait
		    }
		    ros::spinOnce();	
		    
		    if (c >= number_of_taught_categories)
		    {
			c = 1;
		    }
		    else
		    {
			c++;
		    }
		    
		    if (iterations != TPtmp+FNtmp)
		    {
			pub.publish (msg);
			start_time = ros::Time::now();
			while (ros::ok() && (ros::Time::now() - start_time).toSec() < 1)
			{ //wait
			}
			ros::spinOnce();
			ROS_INFO("\t\t[-]- *iterationst= %i", iterations);
			ROS_INFO("\t\t[-]- *TPtmp+FPtmp= %i", TPtmp+FPtmp);
		    }

		    if ( (iterations >= number_of_taught_categories) and (iterations <= window_size * number_of_taught_categories))
		    {    
			if ((TPtmp+FPtmp)!=0)
			    Precision = TPtmp/double (TPtmp+FPtmp);
			else
			    Precision = 0;
		
			monitor_precision (precision_file, Precision);		
			if (Precision > P_Threshold)
			{
			    average_class_precision.push_back(Precision);
			    User_sees_no_improvement_in_precision = false;
			    ROS_INFO("\t\t[-]- Precision= %f", Precision); 
			    double Recall = TPtmp/double (TPtmp+FNtmp);
			    report_current_results(TPtmp,FPtmp,FNtmp,evaluationFile,false);
			    iterations = 1;
			    ros::spinOnce();		
			}  
			    
		    }//if
		    else if ( (iterations >= window_size * number_of_taught_categories)) // In this condition, if we are at iteration I>3n, we only
                                                                                         // compute precision as the average of last 3n, and discart the first
                                                                                         // I-3n iterations.
		    {
			//compute precision of last 3n, and discart the first I-3n iterations
			Precision = compute_precision_of_last_3n (recognition_results, number_of_taught_categories);
			ROS_INFO("\t\t[-]- Precision= %f", Precision);
    
			monitor_precision (precision_file, Precision);				
			report_precision_of_last_3n (evaluationFile, Precision);
			
			Result << "\n\t\t - precision = "<< Precision;
			if (Precision > P_Threshold)
			{
			    average_class_precision.push_back(Precision);
			    User_sees_no_improvement_in_precision = false;
			    report_current_results(TPtmp,FPtmp,FNtmp,evaluationFile,false);
			    ros::spinOnce();		
			    iterations = 1;
			    iterations_user_sees_no_improvment=0;
			} 
			else 
			{
			    iterations_user_sees_no_improvment++;
			    if (iterations_user_sees_no_improvment > user_sees_no_improvment_const)
			    {
				average_class_precision.push_back(Precision);
				User_sees_no_improvement_in_precision = true;
				ROS_INFO("\t\t[-]- User_sees_no_improvement_in_precision");
				ROS_INFO("\t\t[-]- Finish"); 
				ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 
				
				Result.open (evaluationFile.c_str(), std::ofstream::app);
				Result << "\n After " << user_sees_no_improvment_const <<" iterations, user sees no improvement in precision";
				Result.close();
				
				ros::Duration duration = ros::Time::now() - beginProc;
				report_experiment_result (average_class_precision,
							    number_of_instances, 
							    number_of_taught_categories,  
							    evaluationFile, duration);
				category_introduced.close();
				report_all_experiments_results (TP, FP, FN, Obj_Num,			    
								average_class_precision,
								number_of_instances, 
								number_of_taught_categories,
								name_of_approach);
				
				Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);
				
				systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
				ROS_INFO("\t\t[-]- %s", systemStringCommand.c_str()); 
				system( systemStringCommand.c_str());
				return 0 ;
			    }
			}
		    }
		    else
		    {
			
			if ((TPtmp+FPtmp)!= 0)
			    PrecisionSystem = TPtmp/double (TPtmp+FPtmp);
			else 
			    PrecisionSystem=0;
			
			monitor_precision (precision_file, PrecisionSystem);
		    }
		    k++; // k<-k+1 : number of classification result
		    iterations	++;
		}//else	 
	    }
	    	ROS_INFO("\t\t[-]- number of Iterations = %ld", iterations); 
	}
	
	ROS_INFO("\t\t[-]- Finish"); 
	ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 

	//get toc
	ros::Duration duration = ros::Time::now() - beginProc;
	
	report_current_results(TP,FP,FN,evaluationFile,true);
	report_experiment_result (average_class_precision,
			    number_of_instances, 
			    number_of_taught_categories,  
			    evaluationFile, duration);	
	
	category_introduced.close();
	report_all_experiments_results (TP, FP, FN, Obj_Num,			    
				average_class_precision,
				number_of_instances, 
				number_of_taught_categories,
				name_of_approach);
	
	Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);
	systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
	ROS_INFO("\t\t[-]- %s", systemStringCommand.c_str()); 
	system( systemStringCommand.c_str());
	RunCount++;
	

	
//     }//while(RunCount)   
    
    return 0 ;
}



