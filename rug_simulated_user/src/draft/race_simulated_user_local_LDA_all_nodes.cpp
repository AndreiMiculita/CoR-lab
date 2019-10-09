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
#include <local_LDA_based_object_recognition/local_LDA_based_object_recognition_functionality.h>

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
    double spin_image_support_lenght = 0.1;
    int    subsample_spinimages = 10;

    //simulated user parameters
    double P_Threshold = 0.67;  
    int user_sees_no_improvment_const = 100;
    int window_size = 3;
    int number_of_categories =49 ;
    double uniform_sampling_size = 0.03;
    double recognition_threshold = 2;

    int total_number_of_gibbs_sampling_iterations = 50;
    int total_number_of_topic = 20;
    double alpha = 1/*50/total_number_of_topic*/;
    double beta = 0.1;

    std::string name_of_approach = "TPAMI_LDA";

/* _________________________________
  |                                 |
  |         Global Variable         |
  |_________________________________| */

PerceptionDB* _pdb;
typedef pcl::PointXYZRGBA PointT;
vector <SITOV> dictionary ;

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
    
    InstancePathTmp = tmp;
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
	Result << "\n\n\t.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
	Result << "\n\t - Correct classifier - Category Updated";
	Result << "\n\t .<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.\n\n";
	number_of_instances ++;

// 	IntroduceNewInstance (InstancePathTmp, 
// 			      cat_id, track_id, 
// 			      view_id, 
// 			      spin_image_width,
// 			      spin_image_support_lenght,
// 			      subsample_spinimages
// 			      );
	
	
	IntroduceNewInstance2 (home_address,
				InstancePathTmp,
				cat_id, track_id, view_id, 
				spin_image_width,
				spin_image_support_lenght,
				uniform_sampling_size);
    
	//track_id++;
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
// 	IntroduceNewInstance (InstancePathTmp, 
// 			      cat_id, track_id, 
// 			      view_id, 
// 			      spin_image_width,
// 			      spin_image_support_lenght,
// 			      subsample_spinimages
//  			    );

	IntroduceNewInstance2 (home_address,
				InstancePathTmp,
				cat_id, track_id, view_id, 
				spin_image_width,
				spin_image_support_lenght,
				uniform_sampling_size);
	
	//track_id++;
	ROS_INFO ("D1");
	pp.info(std::ostringstream().flush() << "[-]Category Updated******************");
    }
    track_id++;
    Result.close();
    Result.clear();
    pp.printCallback();
}



int main(int argc, char** argv)
{
    int RunCount=1;
    PrettyPrint pp;

       	/* _______________________________________________________________________________
	|                                   						    |
	|  read the dictionary of visual words from race_object_representation directory  |
	|_________________________________________________________________________________| */
// 	/home/hamidreza/Objectrecognition_code/raceua/race_object_representation/ICRA_dictionaries/clusters7048.txt
	string dictionary_path = ros::package::getPath("race_object_representation") + "/dictinary/RGBD/clusters9089.txt";
	dictionary = readClusterCenterFromFile (dictionary_path);
	

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
	/* ________________________________
	|                                 |
	|     Randomly sort categories    |
	|_________________________________| */
 	//generateRrandomSequencesCategories(RunCount);
	
	ros::init (argc, argv, "EVALUATION");
	
	ros::NodeHandle nh;
	_pdb = race_perception_db::PerceptionDB::getPerceptionDB(&nh); //initialize the database class_list_macros
	string name = nh.getNamespace();
	
	/* _____________________________________
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
	nh.param<double>("/perception/uniform_sampling_size", uniform_sampling_size, uniform_sampling_size);
	
        //read simulated teacher parameters
	nh.param<double>("/perception/P_Threshold", P_Threshold, P_Threshold);
	nh.param<int>("/perception/user_sees_no_improvment_const", user_sees_no_improvment_const, user_sees_no_improvment_const);
	nh.param<int>("/perception/window_size", window_size, window_size);	

	//recognition threshold
	nh.param<double>("/perception/recognition_threshold", recognition_threshold, recognition_threshold);
	    
	//read LDA prameters from launch file
	nh.param<int>("/perception/total_number_of_gibbs_sampling_iterations", total_number_of_gibbs_sampling_iterations, total_number_of_gibbs_sampling_iterations);
	nh.param<int>("/perception/total_number_of_topic", total_number_of_topic, total_number_of_topic);
	nh.param<double>("/perception/alpha", alpha, alpha);
	nh.param<double>("/perception/beta", beta, beta);
	
	//TODO : create a function 
	evaluationFile = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count + "/Detail_Evaluation.txt";
	Result.open (evaluationFile.c_str(), std::ofstream::out);
// 	Result <<"\n VS = "<<uniform_sampling_size << ",  IM = "<<spin_image_width<< ",  SL = "<< spin_image_support_lenght << ",  CT = " <<recognition_threshold;
// 	Result <<"\n GS = "<<total_number_of_gibbs_sampling_iterations << ",  NT = "<<total_number_of_topic<< ",  alpha = "<< alpha << ",  beta = " <<beta << "\n\n";
// 	
       string dataset= (home_address == "/home/hamidreza/") ? "IEETA" : "RGB-D Washington";
       Result  << "system configuration:"
		<< "\n\t-experiment_name = " << name_of_approach.c_str()
		<< "\n\t-name_of_dataset = " << dataset
		<< "\n\t-spin_image_width = "<<spin_image_width 
		<< "\n\t-spin_image_support_lenght = "<< spin_image_support_lenght
		<< "\n\t-uniform_sampling_size = "<<uniform_sampling_size
		<< "\n\t-dictionary_size = "<< dictionary.size()
		<< "\n\t-recognition_threshold = "<< recognition_threshold
		<< "\n\t-alpha = "<< alpha
		<< "\n\t-beta = "<< beta
		<< "\n\t-total_number_of_topic = "<< total_number_of_topic
		<< "\n\t-total_number_of_gibbs_sampling_iterations = "<< total_number_of_gibbs_sampling_iterations
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
	
	

		    
	/*___________________________________________________
	 |                                                   |
 	 |   create a subscriber to get recognition feedback |
	 |___________________________________________________| */
	
	unsigned found = name.find_last_of("/\\");
	std::string topic_name = name.substr(0,found) + "/tracking/recognition_result";
	ros::Subscriber sub = nh.subscribe(topic_name, 10, evaluationfunction);

	/* ______________________________
	|                                |
	|         create a publisher     |
	|________________________________| */
	// 
	//std::string pcin_topic = name.substr(0,found) + "/pipeline_default/tracker/tracked_object_point_cloud";  
	//ros::Publisher pub = nh.advertise< race_perception_msgs::PCTOV> (pcin_topic, 1000);
	
	std::string pcin_topic = name.substr(0,found) + "/tracking/recognition_result";  
	ros::Publisher pub = nh.advertise< race_perception_msgs::RRTOV> (pcin_topic, 1000);
// 	                //initialize the Publisher
//                 _p_recognitionResult_publisher = (boost::shared_ptr<ros::Publisher>) new ros::Publisher;
//                 *_p_recognitionResult_publisher = _p_nh->advertise<race_perception_msgs::RRTOV> ("/perception/tracking/recognition_result", 1);
	/* ______________________________
	 |                               |
	 |         Initialization        |
	 |_______________________________| */
	
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

	
		//start tic	
	ros::Time mainBeginProc = ros::Time::now(); 
	ros::Time beginProc = ros::Time::now(); 
      /* _________________________________
	|                                |
	|       wait for 0.5 second      |
	|________________________________| */
	
	ros::Time start_time = ros::Time::now();
	while (ros::ok() && (ros::Time::now() - start_time).toSec() <0.5)
	{  //wait  
	}
	
      /* _________________________________
	|                                |
	|        Introduce category      |
	|________________________________| */

// 	introduceNewCategory( class_index,track_id,instance_number2.at(class_index-1),evaluationFile,
// 			      spin_image_width,
// 			      spin_image_support_lenght,
// 			      subsample_spinimages);
	
	introduceNewCategory2( home_address.c_str(), 
				class_index,track_id,instance_number2.at(class_index-1),evaluationFile,
				spin_image_width,
				spin_image_support_lenght,
				uniform_sampling_size);
	number_of_instances+=3;
	number_of_taught_categories ++;
	category_introduced<< "1\n";

// 	int  track_id = 0;
// 	std::string categoryName;
// 	vector <int> total_number_of_words_in_each_object;
// 	vector <int> total_number_of_words_for_topic;
// 	vector < vector <int> > sample_topic_index ;
// 	vector < vector <int> > object_topic_matrix;
// 	vector < vector <int> > topic_word_matrix;
// 	vector < vector <double> > theta ;
// 	vector < vector <double> > phi;
// 	vector < double > posterior_distribution_topic;
      
	/* _________________________________
	  |                                |
	  |        Simulated Teacher       |
	  |________________________________| */
	
	categoryName = "";    
	float Precision = 0;
	float Recall = 0;
	float F1 =0;
	vector <float> average_class_precision;
	
	while ( class_index < number_of_categories)  // one category already taught above
	{
	    class_index ++; // class index
	    InstancePath= "";
// 	    if (introduceNewCategory( class_index,track_id,instance_number2.at(class_index-1),evaluationFile,
// 			          spin_image_width,
// 				  spin_image_support_lenght,
// 				  subsample_spinimages) == -1)
	     if (introduceNewCategory2(  home_address.c_str(),
					  class_index,track_id,instance_number2.at(class_index-1),evaluationFile,
					  spin_image_width,
					  spin_image_support_lenght,
					  uniform_sampling_size) == -1)
	    {
		ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");
		ros::Duration duration = ros::Time::now() - mainBeginProc;
		report_current_results(TP,FP,FN,evaluationFile,true);
		report_experiment_result ( average_class_precision,
					    number_of_instances, 
					    number_of_taught_categories,  
					    evaluationFile, duration);
		
		report_all_experiments_results (TP, FP, FN, Obj_Num,			    
						average_class_precision,
						number_of_instances, 
						number_of_taught_categories,
						name_of_approach );
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
	    unsigned int c =1;// class index
	    int iterations =1;
	    int iterations_user_sees_no_improvment =0;
	    bool User_sees_no_improvement_in_precision = false; // In the current implementation, If the simulated teacher 
								 // sees the precision doesn't improve in 100 iteration, then, 
								 // it terminares evaluation of the system, originally, 
								 // it was an empirical decision of the human instructor
	    
// 	    while ( ((Precision < P_Threshold ) or (k < number_of_taught_categories)) and (!User_sees_no_improvement_in_precision) )
	    while ( ((F1 < P_Threshold ) or (k < number_of_taught_categories)) 
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
		    ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");

		    ros::Duration duration = ros::Time::now() - mainBeginProc;
		   
		    report_experiment_result (average_class_precision,
					      number_of_instances, 
					      number_of_taught_categories,  
					      evaluationFile, duration);		    		    
		      
		    report_current_results(TP,FP,FN,evaluationFile,true);
  
		    report_all_experiments_results (TP, FP, FN, Obj_Num,			    
						    average_class_precision,
						    number_of_instances, 
						    number_of_taught_categories,
						    name_of_approach );
	
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
		    
		    
		    boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		    pcl::VoxelGrid<PointT > voxelized_point_cloud;	
		    voxelized_point_cloud.setInputCloud (PCDFile);
		    voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
		    voxelized_point_cloud.filter (*target_pc);
	 	    ROS_INFO( "The size of converted point cloud  = %d ", target_pc->points.size() );

		    boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
		    boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
		    keypoint_selection( target_pc, 
					uniform_sampling_size,
					uniform_keypoints,
					uniform_sampling_indices);		    
		    ROS_INFO ("number of keypoints = %i", uniform_keypoints->points.size());
		
		   
		    //Declare a boost share ptr to the spin image msg

		    boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
		    objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
		    pp.info(std::ostringstream().flush() << "Given point cloud has " << target_pc->points.size() << " points.");
		    if (!estimateSpinImages2(target_pc, 
					      0.01 /*downsampling_voxel_size*/, 
					      0.05 /*normal_estimation_radius*/,
					      spin_image_width /*spin_image_width*/,
					      0.0 /*spin_image_cos_angle*/,
					      1 /*spin_image_minimum_neighbor_density*/,
					      spin_image_support_lenght/*spin_image_support_lenght*/,
					      objectViewSpinImages,
					      uniform_sampling_indices))		    
		    {
			ROS_ERROR ("Could not compute spin images");
			return 0;
		    }

		    ROS_INFO( "Computed %d spin images for given point cloud ", objectViewSpinImages->size() );
		
		    /* ___________________________________________
		    |                        	                  |
		    |   compute total number of training object  |
		    |____________________________________________| */
		    		    	
	      //get list of all object categories 
	      vector <ObjectCategory> ListOfObjectCategory = _pdb->getAllObjectCat();
	      //pp.info(std::ostringstream().flush() << ListOfObjectCategory.size()<<" categories exist in the perception database" );
	      vector <int> total_number_of_training_object_in_category;
	      
	      vector < vector <int> > total_number_of_words_in_each_object;
	      vector < vector <int> > total_number_of_words_for_topic;
	      vector < vector < vector <int> > > sample_topic_index ;//initial_topics_for_word_i_object_j
	      vector < vector < vector <int> > > object_topic_matrix;
	      vector < vector < vector <int> > > topic_word_matrix;
	      vector < vector < vector <double> > > theta ;
	      vector < vector < vector <double> > > phi;
	      vector < vector < double > > posterior_distribution_topic;
	      
	      //total number of object must be calculated automatically form train data.
	      for (int i = 0; i < ListOfObjectCategory.size(); ++i) // all category exist in the database 
	      {

		    total_number_of_training_object_in_category.push_back(ListOfObjectCategory.at(i).rtov_keys.size());
		    ROS_INFO("In %s category there are %d training data", ListOfObjectCategory.at(i).cat_name.c_str(), total_number_of_training_object_in_category.at(i));
		    ROS_INFO( "Total_number_of_topic = %d", total_number_of_topic);

		    if (strcmp("zzzzzzzTEST",ListOfObjectCategory.at(i).cat_name.c_str()) ==0)//TODO: it should remove automatically
		      continue;
		      
		    vector <int> total_number_of_words_in_each_object_tmp;
		    vector <int> total_number_of_words_for_topic_tmp;
		    vector < vector <int> > sample_topic_index_tmp ;
		    vector < vector <int> > object_topic_matrix_tmp;
		    vector < vector <int> > topic_word_matrix_tmp;
		    vector < vector <double> > theta_tmp ;
		    vector < vector <double> > phi_tmp;
		    vector < double > posterior_distribution_topic_tmp;
		    
		    std::string category_name = ListOfObjectCategory.at(i).cat_name.c_str() ;
		    /*_________________________________________________
		    |                                		         |
		    |     initialisation of LDA for given category     |
		    |__________________________________________________| */
		    initialisation(  category_name.c_str(), 
				      dictionary, 
				      total_number_of_topic,
				      total_number_of_training_object_in_category.at(i),
				      total_number_of_words_in_each_object_tmp,
				      total_number_of_words_for_topic_tmp,
				      sample_topic_index_tmp,
				      object_topic_matrix_tmp,
				      topic_word_matrix_tmp,
				      posterior_distribution_topic_tmp,
				      theta_tmp,
				      phi_tmp,
				      pp
				    );
			    
		    total_number_of_words_in_each_object.push_back(total_number_of_words_in_each_object_tmp);
		    total_number_of_words_for_topic.push_back(total_number_of_words_for_topic_tmp);
		    sample_topic_index.push_back(sample_topic_index_tmp);
		    object_topic_matrix.push_back(object_topic_matrix_tmp);
		    topic_word_matrix.push_back(topic_word_matrix_tmp);
		    theta.push_back(theta_tmp);
		    phi.push_back(phi_tmp);
		    posterior_distribution_topic.push_back(posterior_distribution_topic_tmp);
		    
// 		    ROS_INFO ("\t[-]LDA initialized for %s category", ListOfObjectCategory.at(i).cat_name.c_str());
// 		    ROS_INFO( "\t[-]theta.at(i).at(0).size() %d",  theta.at(i).at(0).size());
// 		    ROS_INFO( "\t[-]theta.at(i).size() %d",  theta.at(i).size());
// 		    ROS_INFO( "\t[-]theta.at(i).size() %d",  theta.size());
		    
	      }
		    /*_____________
		    |              |
		    |   inference  |
		    |______________| */
		    vector <int> new_total_number_of_words_in_each_object;
		    vector <int> new_total_number_of_words_for_topic;
		    vector < vector <int> > new_sample_topic_index;//TODO: chose a better name of this matrix
		    vector < vector <int> > new_object_topic_matrix;
		    vector < vector <int> > new_topic_word_matrix;
		    vector < vector <double> > new_theta;
		    vector < vector <double> > new_phi;

		    beginProc = ros::Time::now(); 

		    inferenceInitialisation (*objectViewSpinImages,
					      dictionary, 
					      total_number_of_topic,
					      new_total_number_of_words_in_each_object,
					      new_total_number_of_words_for_topic,
					      new_sample_topic_index,//TODO: chose a better name of this matrix
					      new_object_topic_matrix,
					      new_topic_word_matrix,
					      new_theta,
					      new_phi,
					      pp);
		      
		    ros::Duration duration = ros::Time::now() - beginProc;
		    double duration_sec = duration.toSec();
		    ROS_INFO( "Inference initialisation tooks %f secs", duration_sec);
		    
		    vector< double > OCD;
		    for (int i = 0; i < ListOfObjectCategory.size(); ++i) // all category exist in the database 
		    {	    	      
			  std::string category_name = ListOfObjectCategory.at(i).cat_name.c_str() ;

			  //add the spin images of the given test data to database
			  conceptualizeObjectViewSpinImagesInSpecificCategory(category_name.c_str(),1,1000,1,*objectViewSpinImages,pp);
		    
			  vector <int> new_total_number_of_words_for_topic2;
			  for (int j =0; j < total_number_of_words_for_topic.at(i).size(); j++)
			  {
			    new_total_number_of_words_for_topic2.push_back(total_number_of_words_for_topic.at(i).at(j) + new_total_number_of_words_for_topic.at(j));
			  }

			  // add diffrent information of the given test data to the information of the train data for topics inferece
			  //total_number_of_words_in_each_object.push_back(new_total_number_of_words_for_topic.at(0));
			  total_number_of_words_in_each_object.at(i).push_back(new_total_number_of_words_in_each_object.at(0));
			  sample_topic_index.at(i).push_back(new_sample_topic_index.at(0));
			  object_topic_matrix.at(i).push_back(new_object_topic_matrix.at(0));
			  theta.at(i).push_back(new_theta.at(0));
			  phi.at(i).push_back(new_phi.at(0));

			  beginProc = ros::Time::now(); 
			  /*_________________________________________
			  |                                          |
			  |   estimate topics using gibbs sampling   |
			  |__________________________________________| */
			  
			  estimateLDA(  category_name.c_str(),
					total_number_of_gibbs_sampling_iterations,
					alpha /*constant*/,
					beta /*constant*/,
					dictionary, 
					new_total_number_of_words_for_topic2 /*TODO: comput using object_topic_matrix*/,
					total_number_of_words_in_each_object.at(i) /*TODO: comput using object_topic_matrix*/,
					sample_topic_index.at(i),
					object_topic_matrix.at(i),    
					topic_word_matrix.at(i),
					posterior_distribution_topic.at(i),
					theta.at(i),
					phi.at(i),
					pp );
			  
			    duration = ros::Time::now() - beginProc;
			    duration_sec = duration.toSec();
			    ROS_INFO( "topics estimation tooks %f secs", duration_sec);
			    ROS_INFO( "number of topics (theta.at(i).at(0).size())= %d",  theta.at(i).at(0).size());//number of topic
			    ROS_INFO( "number of instances (theta.at(i).size())= %d",  theta.at(i).size());//number of instance
			    ROS_INFO( "number of categories (theta.size())= %d",  theta.size());//number of category
			    
			    //print2DMatrix (  theta.at(i), "Theta");
					    
			    /* __________________________
			      |                          |
			      |      OCD computation     |
			      |__________________________| */
			      beginProc = ros::Time::now(); 

			      vector <double> tmp;
			      for (int k = 0; k < theta.at(i).at(0).size(); ++k)
			      {
				  tmp.push_back(0);
				  
			      } 
			      for (int k = 0; k < theta.at(i).at(0).size(); ++k)
			      {
				  double tmp1 = theta.at(i).at(theta.at(i).size()-1).at(k);
				  tmp.at(k)= tmp1;
				  //printf("%f,",tmp1);
			      } 

      // 			cout<<"\n*********************** Test-representation = [";
      // 			for (int i = 0; i < tmp.size()-1; ++i)
      // 			{
      // 			    cout<< tmp.at(i) <<",";
      // 			}
      // 			cout<< tmp.at(tmp.size()-1)<<"]\n";
			      
			      int match_index= 0;
			      double  maximum_likelihood =1000;
			      //printf("\nsimilarity with %s category = [", ListOfObjectCategory.at(i).cat_name.c_str());
			      for (int k = 0; k < theta.at(i).size()-1; ++k)
			      {
				  double similarity;
				  //KL distance
				   kullbackLiebler2( theta.at(i).at(k), tmp, similarity );  
				  //Euclidean distance
				   //euclidean_distance ( theta.at(i).at(k), tmp, similarity);  
				  //printf ("{%i, %2.4f}, " ,k, similarity); 
							  
				  if ((similarity!=0) and (similarity < maximum_likelihood))
				  {
				      match_index = k;
				      maximum_likelihood = similarity;  
				  } 
			      }
			      //printf ("]\n "); 

			      //create an object_view_key 
			      std::string predicted_view_key = _pdb->makeKey(key::RV, match_index, 1);
			      ROS_INFO("matched_index = %i and ML = %f , OVK = %s", match_index, maximum_likelihood, predicted_view_key.c_str());
			      
			      OCD.push_back(maximum_likelihood);
				  

			      //ROS_INFO( "number of instances before removing test data= %d",  theta.at(i).size());//number of instance

			      //remove the test data form database and train data
			      deleteObjectViewSpinImagesFromSpecificCategory(category_name.c_str(),1,1000,1,objectViewSpinImages);
			      
			      total_number_of_words_in_each_object.at(i).pop_back();
			      sample_topic_index.at(i).pop_back();
			      object_topic_matrix.at(i).pop_back();
			      theta.at(i).pop_back();
			      phi.at(i).pop_back();
		      
		    }

		    int match_index=0;
		    double minimum_OCD = OCD.at(0);
		    ROS_INFO("- SIZE OF OCD = %d", OCD.size());
		    //printf("OCD [%2.4f,", OCD.at(0));
		    for (int i=1; i< OCD.size();++i)
		    {
		      //printf("%2.4f,", OCD.at(i));
		      if (OCD.at(i)<minimum_OCD)
		      {
			minimum_OCD = OCD.at(i);
			match_index = i;
		      }
		    }
		    //printf("------ Match Index =%i\n",match_index);
		    string result;
		    result = ListOfObjectCategory.at(match_index).cat_name.c_str();
		    ROS_INFO("recognition result = %s", result.c_str());

		    RRTOV _rrtov;
		    _rrtov.header.stamp = ros::Time::now();
		    _rrtov.track_id = track_id;
		    _rrtov.view_id =  view_id;
		    _rrtov.recognition_result = result.c_str();
		    _rrtov.ground_truth_name = InstancePath;
		    _rrtov.minimum_distance =minimum_OCD;
			
		    pub.publish (_rrtov);	
	
		    /* _________________________________
		    |                                 |
		    |       wait for 1 second        |
		    |________________________________| */
		    
		    start_time = ros::Time::now();
		    while (ros::ok() && (ros::Time::now() - start_time).toSec() < 0.5)
		    { //wait
		    }
		    ros::spinOnce();	
// 		    
		    if (c >= number_of_taught_categories)
		    {
			c = 1;
		    }
		    else
		    {
			c++;
		    }
		    
// 		    if (iterations != TPtmp+FNtmp)
// 		    {
// 			pub.publish (msg);
// 			start_time = ros::Time::now();
// 			while (ros::ok() && (ros::Time::now() - start_time).toSec() < 1)
// 			{ //wait
// 			}
// 			ros::spinOnce();
// 			ROS_INFO("\t\t[-]- *iterationst= %i", iterations);
// 			ROS_INFO("\t\t[-]- *TPtmp+FPtmp= %i", TPtmp+FPtmp);
// 		    }

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
				ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 
				
				Result.open (evaluationFile.c_str(), std::ofstream::app);
				Result << "\n After " << user_sees_no_improvment_const <<" iterations, user sees no improvement in precision";
				Result.close();

				monitor_precision (local_F1_vs_learned_category, F1);
				monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );

				
				ros::Duration duration = ros::Time::now() - mainBeginProc;
				report_experiment_result (average_class_precision,
							    number_of_instances, 
							    number_of_taught_categories,  
							    evaluationFile, duration);
				category_introduced.close();

				report_current_results(TP,FP,FN,evaluationFile,true);
				
				report_all_experiments_results (TP, FP, FN, Obj_Num,			    
								average_class_precision,
								number_of_instances, 
								number_of_taught_categories,
								name_of_approach);
				
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
	ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 

	//get toc
	ros::Duration duration = ros::Time::now() - mainBeginProc;
	
	monitor_precision (local_F1_vs_learned_category, F1);
	monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );

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



