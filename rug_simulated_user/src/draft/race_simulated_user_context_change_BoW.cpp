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
#include <feature_extraction/spin_image.h>
#include <race_perception_msgs/perception_msgs.h>
#include <object_conceptualizer/object_conceptualization.h>
#include <race_simulated_user/race_simulated_user_functionality.h>
#include <race_perception_utils/print.h>

#include <math.h> 

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
  //spin images parameters
  int    spin_image_width = 4 ;
  double spin_image_support_lenght = 0.2;
  int    subsample_spinimages = 10;

  //simulated user parameters
  double P_Threshold = 0.67;  
  int user_sees_no_improvment_const = 100;
  int window_size = 3;
  int number_of_categories =49 ;
  double uniform_sampling_size = 0.03;
  double recognition_threshold = 2;
  
  //context change parameters
  int number_of_contexts =2;
  double ALC = 38;//21.8; //average number of learned categories (ALC)
  int overlap_between_contexts =0; 
  int total_number_of_experiments = 10;
//   std::string name_of_approach = "Context_Change_BoW";
    std::string name_of_approach = "Context_Change_GOOD";

  vector <SITOV> dictionary ;

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
  std::string evaluationFile, evaluationTable, precision_file, local_F1_vs_learned_category, F1_vs_learned_category;

  std::ofstream Result , Result_table , PrecisionMonitor, local_F1_vs_category, F1_vs_category, NumberofFeedback , category_random, instances_random, category_introduced;
  int TP =0, FP=0, FN=0, TPtmp =0, FPtmp=0, FNtmp=0, Obj_Num=0, number_of_instances=0;//track_id_gloabal = 1 ;//, track_id_gloabal2=1;      

  float PrecisionSystem =0;
  float F1System =0;

  vector <int> recognition_results; // we coded 0: continue(correctly detect unkown object)
				  // 1: TP , 2: FP 
				  //3: FN , 4: FP and FN
				  

void evaluationfunction(const race_perception_msgs::RRTOV &result)
{
    PrettyPrint pp;
    string tmp = result.ground_truth_name.substr(home_address.size(),result.ground_truth_name.size());
    string True_cat = extractCategoryName(tmp);
    InstancePathTmp=tmp;
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
// 	Result << "\n\n\t.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
// 	Result << "\n\t - Correct classifier - Category Updated";
// 	Result << "\n\t .<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.\n\n";
	number_of_instances ++;

// 	IntroduceNewInstance (InstancePathTmp, 
// 			      cat_id, track_id, 
// 			      view_id, 
// 			      spin_image_width,
// 			      spin_image_support_lenght,
// 			      subsample_spinimages
// 			      );
	
	
	IntroduceNewInstanceHistogram ( InstancePathTmp, cat_id, 
					track_id, view_id, pp,
					spin_image_width,
					spin_image_support_lenght,
					subsample_spinimages);
	
	pp.info(std::ostringstream().flush() << "[-]Category Updated");
    }
    else if ((strcmp(True_cat.c_str(),Predict_cat.c_str())!=0))
    {  	
	FP++; FN++;
	FPtmp++; FNtmp++;    
	recognition_results.push_back(4);

	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t1\t1"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
// 	Result << "\n\n\t.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
// 	Result << "\n\t - Correct classifier - Category Updated";
// 	Result << "\n\t .<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.\n\n";	
	number_of_instances++;
// 	IntroduceNewInstance (InstancePathTmp, 
// 			      cat_id, track_id, 
// 			      view_id, 
// 			      spin_image_width,
// 			      spin_image_support_lenght,
// 			      subsample_spinimages
//  			    );

	IntroduceNewInstanceHistogram ( InstancePathTmp, cat_id, 
					track_id, view_id, pp,
					spin_image_width,
					spin_image_support_lenght,
					subsample_spinimages);
	
	
	pp.info(std::ostringstream().flush() << "[-]Category Updated");
    }
    Result.close();
    Result.clear();
    pp.printCallback();
}


int monitorComputationTimeForSpecificPackage (string pakage_name, unsigned int TID, bool init, float duration)
{
  
  
  
  char TIDC [10];
  sprintf( TIDC, "%d",TID );
  std::string processing_time_path = ros::package::getPath(pakage_name)+ "/result/TID"+ TIDC +"processing_time_evaluation.txt";
  std::ofstream processing_time; 
  
  if (init)
  {
      processing_time.open (processing_time_path.c_str(), std::ofstream::out);
      processing_time.precision(8);
      processing_time << "TID" << TID <<" has been initialized at "<< ros::Time::now() <<"\n";
      processing_time.close();
    
  }
  else
  {
      processing_time.open (processing_time_path.c_str(), std::ofstream::app);
      processing_time.precision(8);
      processing_time << duration <<"\n";
      processing_time.close();
  }
  return 0;
}


int main(int argc, char** argv)
{
    int RunCount=1;
    ROS_INFO("Hello world: context_change");
    ros::init (argc, argv, "EVALUATION");
    ros::NodeHandle nh;

    PrettyPrint pp;
    string name = nh.getNamespace();

    
//     while(RunCount <= number_of_categories)
//     {
	/* __________________________________
	|                                   |
	|  Creating a folder for each RUN   |
	|___________________________________| */
	//int to string converting 
	char run_count [10];
	sprintf( run_count, "%d",RunCount );
		
	//Creating a folder for each RUN.
	string systemStringCommand= "mkdir "+ ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
	system( systemStringCommand.c_str());
	
	
// 	for (int i =0; i<5; i++)
// 	{
// 	    int random = rand() % ((int)round(1.25*ALC)-(int)round(0.75*ALC)) + (int)round(0.75*ALC); //ALC = average number of learned categories
// 	    ROS_INFO ("RANDOM = %i", random);
// 	}
// 	
// 	return 0;
	
	/* _______________________________________________________________________________
	|                                   						    |
	|  read the dictionary of visual words from race_object_representation directory  |
	|_________________________________________________________________________________| */
// 	/home/hamidreza/Objectrecognition_code/raceua/race_object_representation/ICRA_dictionaries/clusters7048.txt
// 	string dictionary_path = ros::package::getPath("race_object_representation") + "/dictinary/RGBD/clusters9089.txt";
// 	string dictionary_path = ros::package::getPath("race_object_representation") + "/dictinary/RGBD/clusters9089.txt";
	string dictionary_path = ros::package::getPath("race_object_representation") + "/clusters.txt";

	dictionary = readClusterCenterFromFile (dictionary_path);
	
	//Read dictionaryd
	vector <SITOV> cluster_center = readClusterCenterFromFile (dictionary_path);
	pp.info(std::ostringstream().flush() << "static dictionary has "  << cluster_center.size() <<" words");
	pp.info(std::ostringstream().flush() << "size of each words "  << cluster_center. at(0).spin_image.size() <<" float");
	
	 // initialize perception database 
	_pdb = race_perception_db::PerceptionDB::getPerceptionDB(&nh); //initialize the database class_list_macros
	
	/* _______________________________
	|                                 |
	|     Randomly sort categories    |
	|_________________________________| */
 	generateRrandomSequencesCategories(RunCount);
	
	//TODO : create a function 
	/* ________________________________________
	 |                                       |
	 |     read prameters from launch file   |
	 |_______________________________________| */
	// read database parameter
	nh.param<std::string>("/perception/home_address", home_address, "default_param");
	nh.param<int>("/perception/number_of_categories", number_of_categories, number_of_categories);
	nh.param<std::string>("/perception/name_of_approach", name_of_approach, "default_param");
	
	//read spin images parameters
	nh.param<int>("/perception/spin_image_width", spin_image_width, spin_image_width);
	nh.param<double>("/perception/spin_image_support_lenght", spin_image_support_lenght, spin_image_support_lenght);
	nh.param<int>("/perception/subsample_spinimages", subsample_spinimages, subsample_spinimages);
	
	//read simulated teacher parameters
	nh.param<double>("/perception/P_Threshold", P_Threshold, P_Threshold);
	nh.param<int>("/perception/user_sees_no_improvment_const", user_sees_no_improvment_const, user_sees_no_improvment_const);
	nh.param<int>("/perception/window_size", window_size, window_size);		
	nh.param<int>("/perception/total_number_of_experiments", total_number_of_experiments, total_number_of_experiments);

	//context_change param
	nh.param<double>("/perception/ALC", ALC, ALC); //ALC = average number of learned categories


	int random = rand() % (int)(ceil(0.85*ALC)-floor(0.65*ALC)) + (int)floor(0.65*ALC); //ALC = average number of learned categories

		
	evaluationFile = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count + "/Detail_Evaluation.txt";
	Result.open (evaluationFile.c_str(), std::ofstream::out);
	//Result <<"\nclassification_threshold = " << classification_threshold <<  "\nspin_image_width = " << spin_image_width << "\nsubsample_spinimages = " << subsample_spinimages << "\n\n";
	
	string dataset= (home_address == "/home/hamidreza/") ? "IEETA" : "RGB-D Washington";
//         Result  << "system configuration:"
// 		<< "\n\t-experiment_name = " << name_of_approach.c_str()
// 		<< "\n\t-name_of_dataset = " << dataset
// 		<< "\n\t-spin_image_width = "<<spin_image_width 
// 		<< "\n\t-spin_image_support_lenght = "<< spin_image_support_lenght
// 		<< "\n\t-dictionary_size = "<< dictionary.size()
// 		<< "size of each words "  << cluster_center. at(0).spin_image.size()
// 		<< "\n------------------------------------------------------------------------------------------------------------------------------------\n\n";

	Result  << "system configuration:"
		<< "\n\t-experiment_name = " << name_of_approach.c_str()
		<< "\n\t-name_of_dataset = " << dataset
		<< "\n\t-number_of_bins = 15" 
// 		<< "\n\t-spin_image_width = "<< spin_image_width 
// 		<< "\n\t-spin_image_support_lenght = "<< spin_image_support_lenght
// 		<< "\n\t-uniform_sampling_size = "<< uniform_sampling_size
// 		<< "\n\t-recognition_threshold = "<< recognition_threshold 
		<< "\n\t-index of context change = "<< random
		<< "\n------------------------------------------------------------------------------------------------------------------------------------\n\n";

	Result << "\nNum"<<"\tObject_name" <<"\t\t\t"<< "True_Category" <<"\t\t"<< "Predict_Category"<< "\t"<< "TP" << "\t"<< "FP"<< "\t"<< "FN \t\tDistance";
	Result << "\n------------------------------------------------------------------------------------------------------------------------------------";
	Result.close();
	Result.clear();
	
	precision_file = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count +"/PrecisionMonitor.txt";
	PrecisionMonitor.open (precision_file.c_str(), std::ofstream::trunc);
	PrecisionMonitor.precision(4);
	PrecisionMonitor.close();

	local_F1_vs_learned_category = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count +"/local_F1_vs_learned_category.txt";
	local_F1_vs_category.open (local_F1_vs_learned_category.c_str(), std::ofstream::trunc);
	local_F1_vs_category.precision(4);
	local_F1_vs_category.close();
	
	F1_vs_learned_category = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count +"/F1_vs_learned_category.txt";
	F1_vs_category.open (F1_vs_learned_category.c_str(), std::ofstream::trunc);
	F1_vs_category.precision(4);
	F1_vs_category.close();

	string category_introduced_txt = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count + "/Category_Introduced.txt";
	category_introduced.open (category_introduced_txt.c_str(), std::ofstream::out);
		
	

	ros::Time beginProc = ros::Time::now(); //start tic	
      /* _________________________________
	|                                |
	|       wait for 0.5 second      |
	|________________________________| */
	
	ros::Time start_time = ros::Time::now();
	while (ros::ok() && (ros::Time::now() - start_time).toSec() <0.5)
	{  //wait  
	}
	

	/*___________________________________________________
	 |                                                   |
	 |   create a subscriber to get recognition feedback |
	 |___________________________________________________| */
	
	unsigned found = name.find_last_of("/\\");
	std::string topic_name = name.substr(0,found) + "/tracking/recognition_result";
	ros::Subscriber sub = nh.subscribe(topic_name, 10, evaluationfunction);

	/*________________________________________________________________
	|                                  				    |
	|    create a publisher to publish a point cloud of an object     |
	|_________________________________________________________________| */
	// 
// 	std::string pcin_topic = name.substr(0,found) + "/pipeline_default/tracker/tracked_object_point_cloud";  
// 	ros::Publisher pub = nh.advertise< race_perception_msgs::PCTOV> (pcin_topic, 1);

	std::string topic = name.substr(0,found) + "/pipeline_default/object_representation/histogram_tracked_object_view";  
	ros::Publisher pub_BoW_representation = nh.advertise<race_perception_msgs::CompleteRTOV> (topic, 100);

	

	/*_______________________________
	|                                |
	|         Initialization         |
	|________________________________| */
	
	string categoryName="";
	string InstancePath= "";	
	int class_index = 1; // class index
	int context_index = 0;
		
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
	int number_of_taught_categories_tmp = 0;

	//start tic	
	beginProc = ros::Time::now(); 
	start_time = ros::Time::now();
	vector <unsigned int> c ;
	int c_index = 0;
	/* ______________________________
	|                                |
	|        Introduce category      |
	|________________________________| */
// /*
// 	introduceNewCategory( class_index,track_id,instance_number2.at(class_index-1),evaluationFile,
// 			      spin_image_width,
// 			      spin_image_support_lenght,
// 			      subsample_spinimages);
// 	*/
	
	introduceNewCategoryHistogram(class_index,
				      track_id,
				      instance_number2.at(class_index-1),
				      evaluationFile,
				      pp,
				      spin_image_width,
				      spin_image_support_lenght,
				      subsample_spinimages
 				    );
		
	number_of_instances+=3;
	number_of_taught_categories ++;
	category_introduced<< "1\n";

	c.push_back( class_index );	

	/* _______________________________
	|                                |
	|        Simulated Teacher       |
	|________________________________| */
	
	categoryName = "";    
	float Precision = 0;
	float Recall = 0;
	float F1 =0;
	vector <float> average_class_precision;

	//random number for changing context
	srand(time(NULL));
	//int random = rand() % 20 + 15; // random in the range 15-35 
	//int random = rand() % (int)(ceil(0.85*ALC)-floor(0.65*ALC)) + (int)floor(0.65*ALC); //ALC = average number of learned categories
		
// 	//TODO : define Parameter to choose generate a random number, write a number to the file or read a number from file
// 	string pakage_name = "race_simulated_user";
// 	string file_name= "context_change_idx.txt";
// 	int random = read_a_number_from_file ( pakage_name, file_name);
// 	int randomTmp = random;
// 
// 	if (random==18)
// 	  randomTmp =20;
// 	else 
// 	  randomTmp=18;
// 	write_a_number_to_file ( pakage_name, file_name, randomTmp);
// 	
	
	ROS_INFO ("\t\t[-] BoW - Read random number = %i", random );
	int context_change_itr =0;

	while ( class_index < number_of_categories)  // one category already taught above
	{
	    
	    class_index ++; // class index	
	    ROS_INFO ("\t\t[-] BoW - Read random number = %i", random );

	    /* _______________________________
	    |                                |
	    |         context change         |
	    |________________________________| */
	    
// 	    float random = ((float) rand() / (RAND_MAX)) - 0.3;
// 	    float p_context_change = contextChangeProbability(class_index,number_of_categories_in_dataset);
// 	    ROS_INFO ("\t\t[-] random number = %f, contextChangeProbability = %f", random, p_context_change );

	    //if ((context_index == 0) and random > p_context_change )
	    if ((context_index == 0) and number_of_taught_categories == random )
	    {
	      context_index = 1;
	      ROS_INFO ("\n\n\n\n\n\t\t[-] CONTEXT CHANGE :  random number( %i) < number_of_taught_categories ( %i)\n\n\n\n\n", random, number_of_taught_categories );
	      
	      //class_index = int(number_of_categories/number_of_contexts) * (context_index); 
	      ROS_INFO("class_index = %i", class_index);
	      Result.open (evaluationFile.c_str(), std::ofstream::app);
	      Result <<"\n CONTEXT CHANGE :  random number("<<random << ") < number_of_taught_categories ("<<number_of_taught_categories<< ") \n\n";
	      Result.close();
	      Result.clear();    
	      number_of_taught_categories_tmp = number_of_taught_categories;
	      number_of_taught_categories = 0;
	      context_change_itr = Obj_Num;
	      
	    }

	    c.push_back( class_index );	

	    InstancePath= "";
	          
	    if (introduceNewCategoryHistogram(  class_index,
					    track_id,
					    instance_number2.at(class_index-1),
					    evaluationFile,
					    pp,
					    spin_image_width,
					    spin_image_support_lenght,
					    subsample_spinimages
 					) == -1)
	    {
		ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");
		
		number_of_taught_categories += number_of_taught_categories_tmp;
		ROS_INFO("\t\t[-] number_of_taught_categories : %i", number_of_taught_categories);
		ROS_INFO("\t\t[-] instance_number2.at(class_index-1)= %i", instance_number2.at(class_index-1));
		ROS_INFO("\t\t[-] class_index:%i", class_index);
		
		
		
		ros::Duration duration = ros::Time::now() - beginProc;
		report_current_results(TP,FP,FN,evaluationFile,true);
		report_experiment_result ( average_class_precision,
					    number_of_instances, 
					    number_of_taught_categories,  
					    evaluationFile, duration);
		
// 		report_all_experiments_results (TP, FP, FN, Obj_Num,			    
// 						average_class_precision,
// 						number_of_instances, 
// 						number_of_taught_categories,
// 						name_of_approach );
		
		report_all_context_change_experiments_results (TP, FP, FN, Obj_Num,
								random, context_change_itr,
								average_class_precision,
								number_of_instances, 
								number_of_taught_categories,
								name_of_approach,
								total_number_of_experiments );
		
		
		category_introduced.close();
		
		monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );

		Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);
		Visualize_Local_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, P_Threshold, local_F1_vs_learned_category.c_str());
		Visualize_Global_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, F1_vs_learned_category.c_str());
		Visualize_Number_of_learned_categories_vs_Iterations (RunCount, category_introduced_txt.c_str());
	
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
	    F1=0;
	    Recall=0;
	    unsigned int Ci =0;// Category index
	    
	    int iterations =1;
	    int iterations_user_sees_no_improvment =0;
	    bool User_sees_no_improvement_in_precision = false; // In the current implementation, If the simulated teacher 
								 // sees the precision doesn't improve in 100 iteration, then, 
								 // it terminares evaluation of the system, originally, 
								 // it was an empirical decision of the human instructor
	    
// 	    while ( ((Precision < P_Threshold ) or (k < number_of_taught_categories)) and (!User_sees_no_improvement_in_precision) )
	    while ( ((F1 < P_Threshold ) or (k < number_of_taught_categories)) and (!User_sees_no_improvement_in_precision))
	    {
		//int start_context_index = int(number_of_categories / number_of_contexts) * (context_index);
	    
		ROS_INFO("\t\t[-] number_of_taught_categories in this context : %i", number_of_taught_categories );
		//ROS_INFO("\t\t[-] TEST");

		
		if (class_index > number_of_categories)
		{
		    number_of_taught_categories += number_of_taught_categories_tmp;
		    ROS_INFO ("Note: the experiment is terminated because class_index is larger than number of categories data");	
		    ROS_INFO("\t\t[-] number_of_taught_categories : %i", number_of_taught_categories );
		    ROS_INFO("\t\t[-] class_index:%i", class_index);

		    ros::Duration duration = ros::Time::now() - beginProc;
		    report_current_results(TP,FP,FN,evaluationFile,true);
		    report_experiment_result ( average_class_precision,
						number_of_instances, 
						number_of_taught_categories,  
						evaluationFile, duration);
		    
// 		    report_all_experiments_results (TP, FP, FN, Obj_Num,			    
// 						    average_class_precision,
// 						    number_of_instances, 
// 						    number_of_taught_categories,
// 						    name_of_approach );
		    
		    report_all_context_change_experiments_results (TP, FP, FN, Obj_Num,
								random,context_change_itr,
								average_class_precision,
								number_of_instances, 
								number_of_taught_categories,
								name_of_approach,
								total_number_of_experiments );
		    
		    category_introduced.close();
		    
		    monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );

		    Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);
		    Visualize_Local_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, P_Threshold, local_F1_vs_learned_category.c_str());
		    Visualize_Global_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, F1_vs_learned_category.c_str());
		    Visualize_Number_of_learned_categories_vs_Iterations (RunCount, category_introduced_txt.c_str());
	    
		    systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
		    system( systemStringCommand.c_str());
		    return (0) ;
		}
		
		if (Ci > number_of_taught_categories+number_of_taught_categories_tmp-1)
		{
		    Ci = 0;
		}
		
		//previous approach with overlap_between_contexts
// 		if (!inContext(c.at(Ci), context_index, number_of_contexts, overlap_between_contexts, number_of_categories))
// 		{
// 		    Ci++;
// 		    continue;
// 		}

		
		if (!inContext(c.at(Ci), context_index, random, number_of_categories ))
		{
		    Ci++;
		    continue;
		}

		
		
		ROS_INFO("\t\t[-] number_of_taught_categories : %i", number_of_taught_categories);
		ROS_INFO("\t\t[-] Ci:%i",Ci);
		ROS_INFO("\t\t[-] test");
		ROS_INFO("\t\t[-] Size of c = %i", c.size());
		ROS_INFO("\t\t[-] class_index:%i", c.at(Ci));
		ROS_INFO("\t\t[-] class_index:%i", class_index);

	   	ROS_INFO("\t\t[-] context_index:%i",context_index);
	   	//ROS_INFO("\t\t[-] class_index:%i",class_index);
				
		category_introduced<< "0\n";
// 		ROS_INFO("\t\t[-] Iteration:%i",iterations);
		ROS_INFO("\t\t[-] Class index :%i",c.at(Ci));
		ROS_INFO("\t\t[-] Ci:%i",Ci);
		ROS_INFO("\t\t[-] Instance number:%i",instance_number2.at(c.at(Ci)-1));
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
		selectAnInstancefromSpecificCategory(c.at(Ci), instance_number2.at(c.at(Ci)-1), InstancePath);
		
		ROS_INFO("\t\t[-]-Test Instance: %s", InstancePath.c_str());
    
		// check the selected instance exist or not? if yes, send it to the race_feature_extractor
		if (InstancePath.size() < 2) 
		{
		    ROS_INFO("\t\t[-]-The %s file does not exist", InstancePath.c_str());
		    category_introduced.close();
			    
		    number_of_taught_categories += number_of_taught_categories_tmp;
		    ROS_INFO("\t\t[-]- total number of taught categories= %i", number_of_taught_categories); 
		    ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");

		    ros::Duration duration = ros::Time::now() - beginProc;
		   
		    report_experiment_result (average_class_precision,
					      number_of_instances, 
					      number_of_taught_categories,  
					      evaluationFile, duration);		    		    
		      
		    report_current_results(TP,FP,FN,evaluationFile,true);
  
// 		    report_all_experiments_results (TP, FP, FN, Obj_Num,			    
// 						    average_class_precision,
// 						    number_of_instances, 
// 						    number_of_taught_categories,
// 						    name_of_approach );

		    report_all_context_change_experiments_results (TP, FP, FN, Obj_Num,
								random,context_change_itr,
								average_class_precision,
								number_of_instances, 
								number_of_taught_categories,
								name_of_approach,
								total_number_of_experiments );
		    
		    monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );
		    
		    Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);
		    Visualize_Local_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, P_Threshold, local_F1_vs_learned_category.c_str());
		    Visualize_Global_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, F1_vs_learned_category.c_str());
		    Visualize_Number_of_learned_categories_vs_Iterations (RunCount, category_introduced_txt.c_str());

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
		    
// 		    boost::shared_ptr<PointCloud<PointT> > cloud_filtered (new PointCloud<PointT>);
// 		    pcl::VoxelGrid<PointT > voxelized_point_cloud;	
// 		    voxelized_point_cloud.setInputCloud (PCDFile);
// 		    voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
// 		    voxelized_point_cloud.filter (*cloud_filtered);
// 		    
// 		    //Declare PCTOV msg 
// 		    boost::shared_ptr<race_perception_msgs::PCTOV> msg (new race_perception_msgs::PCTOV );
// 		    pcl::toROSMsg(*cloud_filtered, msg->point_cloud);
// 		    msg->track_id = track_id;//it is 
// 		    msg->view_id = view_id;		    
// 		    msg->ground_truth_name = InstancePath;//extractCategoryName(InstancePath);
// 		    pub.publish (msg);
// 		    ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", InstancePath.c_str());
// 		    
		    
		    
		    /* _____________________________________________________
		    |                                                     |
		    |  Compute the Spin-Images for the given test object  |
		    |_____________________________________________________| */
		
		    boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		    pcl::VoxelGrid<PointT > voxelized_point_cloud;	
		    voxelized_point_cloud.setInputCloud (PCDFile);
		    voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
		    voxelized_point_cloud.filter (*target_pc);
		    ROS_INFO( "The size of converted point cloud  = %d ", target_pc->points.size() );
		    
		    //Declare a boost share ptr to the spin image msg		  
		    boost::shared_ptr< vector <SITOV> > testObjectViewSpinImages;
		    testObjectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
		    
		    boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
		    boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
		    keypoint_selection( target_pc, 
					  uniform_sampling_size,
					  uniform_keypoints,
					  uniform_sampling_indices);
		      
		    ROS_INFO ("uniform_sampling_size = %f", uniform_sampling_size);
		    ROS_INFO ("number of keypoints = %i", uniform_keypoints->points.size());
		    
		    
		    if (!estimateSpinImages2(target_pc, 
					    0.01 /*downsampling_voxel_size*/, 
					    0.05 /*normal_estimation_radius*/,
					    spin_image_width /*spin_image_width*/,
					    0.0 /*spin_image_cos_angle*/,
					    1 /*spin_image_minimum_neighbor_density*/,
					    spin_image_support_lenght /*spin_image_support_lenght*/,
					    testObjectViewSpinImages,
					    uniform_sampling_indices /*subsample spinimages*/))
		    {
			pp.error(std::ostringstream().flush() << "Could not compute spin images");
			return (0);
		    }
		    pp.info(std::ostringstream().flush() << "Computed " << testObjectViewSpinImages->size() << " spin images for given point cloud. ");
		    
		    SITOV object_representation;
		    objectRepresentationBagOfWords (dictionary, *testObjectViewSpinImages, object_representation);

		    
		    
		    		    
// 		    //Declare a boost share ptr to the spin image msg
// 		    boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
// 		    objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
// 		    pp.info(std::ostringstream().flush() << "Given point cloud has " << PCDFile->points.size() << " points.");
// 	  
// 		    if (!estimateSpinImages( PCDFile, 
// 				0.01 /*downsampling_voxel_size*/, 
// 				0.05 /*normal_estimation_radius*/,
// 				spin_image_width /*spin_image_width*/,
// 				0.0 /*spin_image_cos_angle*/,
// 				1 /*spin_image_minimum_neighbor_density*/,
// 				spin_image_support_lenght /*spin_image_support_lenght*/,
// 				objectViewSpinImages,
// 				subsample_spinimages /*subsample spinimages*/
// 				))
// 		    {
// 
// 			ROS_INFO( "Could not compute spin images");
// 			return 1;
// 		    }
// 		    
// 
// 		    /* ____________________________________________________
// 		      |                                                  |
// 		      |  Representing object by the static dictionary    |
// 		      |__________________________________________________| */
// 	     
// 
// 		      SITOV object_representation;
// 		      objectRepresentationBagOfWords (cluster_center, *objectViewSpinImages, object_representation);

		      //ROS_INFO(" [-] size of object view histogram %ld",object_representation.spin_image.size());
		      pp.info(std::ostringstream().flush() << "size of object view histogram " << object_representation.spin_image.size() );
		      
		      //Declare SITOV (Spin Images of Tracked Object View)
		      SITOV _sitov;

		      //Declare RTOV (Representation of Tracked Object View)
		      RTOV _rtov;
		      _rtov.track_id = track_id;
		      _rtov.view_id = view_id;
		      
		      //declare the RTOV complete variable
		      race_perception_msgs::CompleteRTOV _crtov;
		      _crtov.track_id = track_id;
		      _crtov.view_id = view_id;
		      _crtov.ground_truth_name = InstancePath.c_str();

		      //Add the object view representation in msg_out to put in the DB
		      _sitov = object_representation; //copy spin images
		      _sitov.track_id = track_id; //copy track_id
		      _sitov.view_id = view_id; //copy view_id
		      _sitov.spin_img_id = 1; //copy spin image id

		      //Addd sitov to completertov sitov list
		      _crtov.sitov.push_back(_sitov);

		      //Publish the CompleteRTOV to recognition
		      pub_BoW_representation.publish (_crtov);
		    
		    
		    
		    
		    Ci++;

		    /* _________________________________
		    |                                |
		    |       wait for 1 second        |
		    |________________________________| */
		    
		    start_time = ros::Time::now();
		    while (ros::ok() && (ros::Time::now() - start_time).toSec() < 1)
		    { //wait
		    }
		    ros::spinOnce();	

		    if ( (iterations >= number_of_taught_categories) and (iterations <= window_size * number_of_taught_categories))
		    {    
			if ((TPtmp+FPtmp)!=0)
			{
			    Precision = TPtmp/double (TPtmp+FPtmp);
			}
			else
			{
			    Precision = 0;
			}		
			if ((TPtmp+FNtmp)!=0)
			{
			    Recall = TPtmp/double (TPtmp+FNtmp);
			}
			else
			{
			    Recall = 0;
			}
			if ((Precision + Recall)!=0)
			{
			      F1 = 2 * (Precision * Recall )/(Precision + Recall );
			}
			else
			{
			    F1 = 0;
			}
			    
// 			monitor_precision (precision_file, Precision);		
			monitor_precision (precision_file, F1);		

// 			if (Precision > P_Threshold)
			if (F1 > P_Threshold)
			{
// 			    average_class_precision.push_back(Precision);
			    average_class_precision.push_back(F1);
			    User_sees_no_improvement_in_precision = false;
			    ROS_INFO("\t\t[-]- Precision= %f", Precision);
			    ROS_INFO("\t\t[-]- F1 = %f", F1); 
// 			    double Recall = TPtmp/double (TPtmp+FNtmp);
			    report_current_results(TPtmp,FPtmp,FNtmp,evaluationFile,false);
			    iterations = 1;
			    monitor_precision (local_F1_vs_learned_category, F1);
			    ros::spinOnce();		
			}  
			    
		    }//if
		    else if ( (iterations > window_size * number_of_taught_categories)) // In this condition, if we are at iteration I>3n, we only
                                                                                         // compute precision as the average of last 3n, and discart the first
                                                                                         // I-3n iterations.
		    {
			//compute precision of last 3n, and discart the first I-3n iterations
// 			Precision = compute_precision_of_last_3n (recognition_results, number_of_taught_categories);
			F1 = compute_Fmeasure_of_last_3n (recognition_results, number_of_taught_categories);
// 			ROS_INFO("\t\t[-]- Precision= %f", Precision);
    
// 			monitor_precision (precision_file, Precision);
			monitor_precision (precision_file, F1);		
			
// 			report_precision_of_last_3n (evaluationFile, Precision);
			report_F1_of_last_3n (evaluationFile, F1);
			
// 			Result << "\n\t\t - precision = "<< Precision;
			Result << "\n\t\t - F1 = "<< F1;
// 			if (Precision > P_Threshold)
			if (F1 > P_Threshold)  
			{
// 			    average_class_precision.push_back(Precision);
			    average_class_precision.push_back(F1);

			    User_sees_no_improvement_in_precision = false;
			    report_current_results(TPtmp,FPtmp,FNtmp,evaluationFile,false);
			    monitor_precision (local_F1_vs_learned_category, F1);
			    iterations = 1;
			    iterations_user_sees_no_improvment=0;
			    ros::spinOnce();		
			} 
			else 
			{
			    iterations_user_sees_no_improvment++;
			    ROS_INFO("\t\t[-]- %i user_sees_no_improvement_in_F1", iterations_user_sees_no_improvment);

			    if (iterations_user_sees_no_improvment > user_sees_no_improvment_const)
			    {
// 				average_class_precision.push_back(Precision);
				average_class_precision.push_back(F1);

				User_sees_no_improvement_in_precision = true;
				ROS_INFO("\t\t[-]- User_sees_no_improvement_in_precision");
				ROS_INFO("\t\t[-]- Finish"); 

				number_of_taught_categories += number_of_taught_categories_tmp;
				ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 
				
				Result.open (evaluationFile.c_str(), std::ofstream::app);
				Result << "\n After " << user_sees_no_improvment_const <<" iterations, user sees no improvement in precision";
				Result.close();

				monitor_precision (local_F1_vs_learned_category, F1);
				monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );

				
				ros::Duration duration = ros::Time::now() - beginProc;
				report_experiment_result (average_class_precision,
							    number_of_instances, 
							    number_of_taught_categories,  
							    evaluationFile, duration);
				category_introduced.close();

				report_current_results(TP,FP,FN,evaluationFile,true);
				
// 				report_all_experiments_results (TP, FP, FN, Obj_Num,			    
// 								average_class_precision,
// 								number_of_instances, 
// 								number_of_taught_categories,
// 								name_of_approach);
				report_all_context_change_experiments_results (TP, FP, FN, Obj_Num,
								random,context_change_itr,
								average_class_precision,
								number_of_instances, 
								number_of_taught_categories,
								name_of_approach,
								total_number_of_experiments );
				
				Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);
				Visualize_Local_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, P_Threshold, local_F1_vs_learned_category.c_str());
				Visualize_Global_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, F1_vs_learned_category.c_str());
				Visualize_Number_of_learned_categories_vs_Iterations (RunCount, category_introduced_txt.c_str());
				
				systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
				ROS_INFO("\t\t[-]- %s", systemStringCommand.c_str()); 
				system( systemStringCommand.c_str());
				
				
				return 0 ;
			    }
			}
		    }
		    else
		    {
			
// 			if ((TPtmp+FPtmp)!= 0)
// 			    PrecisionSystem = TPtmp/double (TPtmp+FPtmp);
// 			else 
// 			    PrecisionSystem=0;
				
			if ((TPtmp+FPtmp)!=0)
			{
			    Precision = TPtmp/double (TPtmp+FPtmp);
			}
			else
			{
			    Precision = 0;
			}		
			if ((TPtmp+FNtmp)!=0)
			{
			    Recall = TPtmp/double (TPtmp+FNtmp);
			}
			else
			{
			    Recall = 0;
			}
			if ((Precision + Recall)!=0)
			{
			      F1System = 2 * (Precision * Recall )/(Precision + Recall );
			}
			else
			{
			    F1System = 0;
			}
			
			monitor_precision (precision_file, F1System);
		    }
		    k++; // k<-k+1 : number of classification result
		    iterations	++;
		}//else	 
	    }
// 	    	ROS_INFO("\t\t[-]- number of Iterations = %ld", iterations);
		monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );
	}
	
	ROS_INFO("\t\t[-]- Finish"); 

// 	number_of_taught_categories += number_of_taught_categories_tmp;
	ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 

	//get toc
	ros::Duration duration = ros::Time::now() - beginProc;
	
	monitor_precision (local_F1_vs_learned_category, F1);
	monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );

	report_current_results(TP,FP,FN,evaluationFile,true);
	report_experiment_result (average_class_precision,
			    number_of_instances, 
			    number_of_taught_categories,  
			    evaluationFile, duration);	
	
	category_introduced.close();
// 	report_all_experiments_results (TP, FP, FN, Obj_Num,			    
// 				average_class_precision,
// 				number_of_instances, 
// 				number_of_taught_categories,
// 				name_of_approach);
	
	report_all_context_change_experiments_results (TP, FP, FN, Obj_Num,
							random,context_change_itr,
							average_class_precision,
							number_of_instances, 
							number_of_taught_categories,
							name_of_approach,
							total_number_of_experiments );
	
	Visualize_simulated_teacher_in_MATLAB(RunCount, P_Threshold, precision_file);
	Visualize_Local_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, P_Threshold, local_F1_vs_learned_category.c_str());
	Visualize_Global_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, F1_vs_learned_category.c_str());
	Visualize_Number_of_learned_categories_vs_Iterations (RunCount, category_introduced_txt.c_str());
	systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
	ROS_INFO("\t\t[-]- %s", systemStringCommand.c_str()); 
	system( systemStringCommand.c_str());
	RunCount++;
		


	
//     }//while(RunCount)   
    
    return 0 ;
}



