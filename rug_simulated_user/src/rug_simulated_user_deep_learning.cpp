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

#include <object_descriptor/object_descriptor_functionality.h>

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
#include <rug_simulated_user/rug_simulated_user_functionality.h>
#include <race_perception_utils/print.h>
#include <rug_kfold_cross_validation/rug_kfold_cross_validation_functionality.h>


// #include <race_deep_learning_feature_extraction/vgg16_model.h>
#include <race_deep_learning_feature_extraction/deep_representation.h>


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
  string home_address = "/home/cor/datasets/washington_RGBD_object/"; 
  	
	//spin images parameters
  int    spin_image_width = 8 ;
  double spin_image_support_lenght = 0.2;
  int    subsample_spinimages = 10;
 double downsampling_voxel_size = 0.001;

  //simulated user parameters
  double protocol_threshold = 0.67;  
  int user_sees_no_improvment_const = 100;
  int window_size = 3;
  int number_of_categories =51 ;
  double uniform_sampling_size = 0.03;
  double recognition_threshold = 200;
  
  //deep learning parameters
  bool downsampling = false;
  bool image_normalization = true;
  bool multiviews = true;
  bool max_pooling = true;
  bool modelnet_dataset = false; // we do not need to perform PCA for modelNet dataset

  //distance function
  string distance_function="euclidean";

  std::string deep_learning_architecture = "/vgg16_service";

  std::string name_of_approach = "deep_learning_architectures_and_GOOD";

  // because of updating memory in evaluation fuction, deep_learning_server should be gloabal.
  ros::ServiceClient deep_learning_server;
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

  
  int  off_line_flag  = 1;
  int  number_of_bins = 5;
  int  adaptive_support_lenght = 1;
  double global_image_width =0.5;
  int sign = 1;
  int threshold = 10;	 

  
  std::string evaluation_file, evaluationTable, precision_file, local_f1_vs_learned_category, f1_vs_learned_category;

  std::ofstream summary_of_experiment , PrecisionMonitor, local_f1_vs_category, f1_vs_category, NumberofFeedback , category_random, instances_random, category_introduced;
  int TP = 0, FP= 0, FN = 0, tp_tmp = 0, fp_tmp = 0, fn_tmp = 0, obj_num = 0, number_of_instances = 0;


  vector <int> recognition_results; // we coded 0: continue(correctly detect unkown object)
									// 1: TP , 2: FP , 3: FN , 4: FP and FN
				  


void evaluationFunction(const race_perception_msgs::RRTOV &result)
{
    PrettyPrint pp;
    ROS_INFO("ground_truth_name = %s", result.ground_truth_name.c_str());
    string instance_path = result.ground_truth_name.substr(home_address.size(),result.ground_truth_name.size());
	string tmp = instance_path;
	string true_cat = extractCategoryName(tmp);

    std:: string object_name;
    object_name = extractObjectNameSimulatedUser (result.ground_truth_name);
    pp.info(std::ostringstream().flush() << "extractObjectName: " << object_name.c_str()); 
    
    obj_num++;
    pp.info(std::ostringstream().flush() << "track_id="<< result.track_id << "\tview_id=" << result.view_id);
          
    float minimum_distance = result.minimum_distance;
 
    string predicted_cat;
    predicted_cat= result.recognition_result.c_str();   
    pp.info(std::ostringstream().flush() << "[-]object_name: "<< object_name.c_str());
    pp.info(std::ostringstream().flush() << "[-]true_category: "<<true_cat.c_str());
    pp.info(std::ostringstream().flush() << "[-]predicted_category: " << predicted_cat.c_str());
	true_cat = fixedLength (true_cat , 15); // to have a pretty report, while the length of given string is less than 15, we add space at the end of the string
	predicted_cat = fixedLength (predicted_cat , 15); // to have a pretty report, while the length of given string is less than 15, we add space at the end of the string

    char unknown[] = "unknown";
   
    summary_of_experiment.open( evaluation_file.c_str(), std::ofstream::app);    
    summary_of_experiment.precision(3);
   
    if ((strcmp(true_cat.c_str(),unknown)==0) && (strcmp(predicted_cat.c_str(),unknown)==0))
    { 	
		recognition_results.push_back(0);// continue
		summary_of_experiment << "\n"<<obj_num<<"\t"<<object_name <<"\t\t"<< true_cat <<"\t\t"<< predicted_cat <<"\t\t"<< "0\t0\t0"<< "\t\t"<< minimum_distance;
		summary_of_experiment << "\n----------------------------------------------------------------------------------------------------------------------------------------";
    }
    else if ((strcmp(true_cat.c_str(),predicted_cat.c_str())==0))
    { 
		TP++;
		tp_tmp++;
		recognition_results.push_back(1);
		
		summary_of_experiment << "\n"<<obj_num<<"\t"<<object_name <<"\t\t"<< true_cat <<"\t\t"<< predicted_cat <<"\t\t"<< "1\t0\t0" << "\t\t"<< minimum_distance;
		summary_of_experiment << "\n----------------------------------------------------------------------------------------------------------------------------------------";
    }
    else if ((strcmp(true_cat.c_str(),unknown)==0) && (strcmp(predicted_cat.c_str(),unknown)!=0))
    { 	
		FP++; 
		fp_tmp++;
		recognition_results.push_back(2);

		summary_of_experiment << "\n"<<obj_num<<"\t"<<object_name <<"\t\t"<< true_cat <<"\t\t"<< predicted_cat <<"\t\t"<< "0\t1\t0"<< "\t\t"<< minimum_distance;
		summary_of_experiment << "\n----------------------------------------------------------------------------------------------------------------------------------------";
    }
    else if ((strcmp(true_cat.c_str(),unknown)!=0) && (strcmp(predicted_cat.c_str(),unknown)==0))
    { 	
		FN++;
		fn_tmp++;
		recognition_results.push_back(3);

		summary_of_experiment << "\n"<<obj_num<<"\t"<<object_name <<"\t\t"<< true_cat <<"\t\t"<< predicted_cat <<"\t\t"<< "0\t0\t1"<< "\t\t"<< minimum_distance;
		summary_of_experiment << "\n========================================================================================================================================";

		number_of_instances ++;
	
		IntroduceNewInstanceDeepLearningUsingGOOD ( home_address,
													instance_path,
													cat_id, track_id, view_id,
													adaptive_support_lenght,
													global_image_width,
													threshold,
													number_of_bins,
													deep_learning_server,
													pp );
		
		track_id++;
		pp.info(std::ostringstream().flush() << "[-]Category Updated");
    }
    else if ((strcmp(true_cat.c_str(),predicted_cat.c_str())!=0))
    {  	
		FP++; FN++;
		fp_tmp++; fn_tmp++;    
		recognition_results.push_back(4);

		summary_of_experiment << "\n"<<obj_num<<"\t"<<object_name <<"\t\t"<< true_cat <<"\t\t"<< predicted_cat <<"\t\t"<< "0\t1\t1"<< "\t\t"<< minimum_distance;
		summary_of_experiment << "\n========================================================================================================================================";

		number_of_instances++;

		IntroduceNewInstanceDeepLearningUsingGOOD ( home_address,
													instance_path,
													cat_id, track_id, view_id,
													adaptive_support_lenght,
													global_image_width,
													threshold,
													number_of_bins,
													deep_learning_server,
													pp );

		track_id++;
		pp.info(std::ostringstream().flush() << "[-]Category Updated");
    }
    summary_of_experiment.close();
    summary_of_experiment.clear();
    pp.printCallback();
}

int main(int argc, char** argv)
{
   
	/* __________________________________
	|                                   |
	|  Creating a folder for each RUN   |
	|___________________________________| */
	int experiment_number = 1;
	string system_command= "mkdir "+ ros::package::getPath("rug_simulated_user")+ "/result/experiment_1";
	system( system_command.c_str());

	PrettyPrint pp; // pp stands for pretty print


	precision_file = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/PrecisionMonitor.txt";
	PrecisionMonitor.open (precision_file.c_str(), std::ofstream::trunc);
	PrecisionMonitor.precision(4);
	PrecisionMonitor.close();

	local_f1_vs_learned_category = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/local_f1_vs_learned_category.txt";
	local_f1_vs_category.open (local_f1_vs_learned_category.c_str(), std::ofstream::trunc);
	local_f1_vs_category.precision(4);
	local_f1_vs_category.close();
	
	f1_vs_learned_category = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/f1_vs_learned_category.txt";
	f1_vs_category.open (f1_vs_learned_category.c_str(), std::ofstream::trunc);
	f1_vs_category.precision(4);
	f1_vs_category.close();
		
	ros::init (argc, argv, "EVALUATION");
	ros::NodeHandle nh;
	
	// initialize perception database 
	_pdb = race_perception_db::PerceptionDB::getPerceptionDB(&nh); //initialize the database class_list_macros
	string name = nh.getNamespace();
	
	/* _____________________________________
	|                                       |
	|    read prameters from launch file    |
	|_______________________________________| */

	
	// read database parameter
	nh.param<std::string>("/perception/home_address", home_address, home_address);
	nh.param<int>("/perception/number_of_categories", number_of_categories, number_of_categories);
	nh.param<std::string>("/perception/name_of_approach", name_of_approach, name_of_approach);
  
    // read distance function
    nh.param<std::string>("/perception/distance_function", distance_function, distance_function);

    //parameters of GOOD object descriptor
    nh.param<int>("/perception/number_of_bins", number_of_bins, number_of_bins);
    nh.param<double>("/perception/global_image_width", global_image_width, global_image_width);		

    nh.param<int>("/perception/adaptive_support_lenght", adaptive_support_lenght, adaptive_support_lenght);
    nh.param<int>("/perception/threshold", threshold, threshold);
    
    //recognition threshold
    nh.param<double>("/perception/recognition_threshold", recognition_threshold, recognition_threshold);

    // read deep_learning_architecture parameter
    nh.param<std::string>("/perception/deep_learning_architecture", deep_learning_architecture, deep_learning_architecture);

    // read image_normalization parameter 0 = FLASE, 1 = TRUE 
    nh.param<bool>("/perception/image_normalization", image_normalization, image_normalization);
    string image_normalization_flag = (image_normalization == 0) ? "FALSE" : "TRUE";

   // read multiviews parameter 0 = FLASE, 1 = TRUE 
    nh.param<bool>("/perception/multiviews", multiviews, multiviews);
    string multiviews_flag = (multiviews == 0) ? "FALSE" : "TRUE";

   // read max_pooling parameter 0 = FLASE, 1 = TRUE 
    nh.param<bool>("/perception/max_pooling", max_pooling, max_pooling);
    string pooling_flag = (max_pooling == 0) ? "Avg" : "Max";

    // read downsampling parameter 0 = FLASE, 1 = TRUE 
    nh.param<bool>("/perception/downsampling", downsampling, downsampling);
    string downsampling_flag = (downsampling == 0) ? "FALSE" : "TRUE";
    // read downsampling_voxel_size parameter
    nh.param<double>("/perception/downsampling_voxel_size", downsampling_voxel_size, downsampling_voxel_size);
    downsampling_voxel_size = downsampling_voxel_size * 0.02;
    //set_parameters (name_of_approach);

    // read modelnet_dataset parameter 0 = FLASE, 1 = TRUE 
    nh.param<bool>("/perception/modelnet_dataset", modelnet_dataset, modelnet_dataset);
    string modelnet_dataset_flag = (modelnet_dataset == 0) ? "False" : "True";


	//read simulated teacher parameters
	nh.param<double>("/perception/protocol_threshold", protocol_threshold, protocol_threshold);
	nh.param<int>("/perception/user_sees_no_improvment_const", user_sees_no_improvment_const, user_sees_no_improvment_const);
	nh.param<int>("/perception/window_size", window_size, window_size);	

	//recognition threshold
	nh.param<double>("/perception/recognition_threshold", recognition_threshold, recognition_threshold);

	string dataset= (home_address == "/home/cor/datasets/restaurant_object_dataset/") ? "Restaurant Object Dataset" : "RGB-D Washington";

	evaluation_file = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/summary_of_experiment_vgg16_downsampling" + downsampling_flag + "_" + (downsampling ? "voxel_size" + std::to_string(downsampling_voxel_size) : "") + "_" +  distance_function + "_" + std::to_string(number_of_bins) + "bins.txt";
	summary_of_experiment.open (evaluation_file.c_str(), std::ofstream::out);
	
	summary_of_experiment  <<"system configuration:" 
			<< "\n\t-experiment_name = " << name_of_approach
			<< "\n\t-name_of_dataset = " << dataset 
			//<< "\n\t-down_sampling_resolution = "<< 0.001* (exp_num)
			<< "\n\t-number_of_bins = "<< number_of_bins
			<< "\n\t-number_of_category = "<< "51"
			<< "\n\t-name_of_network = " << deep_learning_architecture
			<< "\n\t-downsampling = " << downsampling_flag
			<< "\n\t-downsampling_voxel_size = " << downsampling_voxel_size
			<< "\n\t-image_normalization = " << image_normalization_flag
			<< "\n\t-multiviews = " << multiviews_flag
			<< "\n\t-pooling = " << pooling_flag
			<< "\n\t-modelnet_dataset = " << modelnet_dataset_flag 
			<< "\n\t-simulated_user_threshold = " << protocol_threshold 
			
			<< "\n------------------------------------------------------------------------------------------------------------------------------------\n\n";

	summary_of_experiment << "\n\nNo."<<"\tobject_name" <<"\t\t\t\t"<< "ground_truth" <<"\t\t"<< "prediction"<< "\t\t"<< "TP" << "\t"<< "FP"<< "\t"<< "FN \t\tdistance";
	summary_of_experiment << "\n-----------------------------------------------------------------------------------------------------------------------------------------";
	summary_of_experiment.close();


	/* _______________________________
	|                                 |
	|     Randomly sort categories    |
	|_________________________________| */
 	generateRrandomSequencesCategories(experiment_number);

	string category_introduced_txt = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/Category_Introduced.txt";
	category_introduced.open (category_introduced_txt.c_str(), std::ofstream::out);

	/* _____________________________
    |                               |
    |     create a client server    |
    |_______________________________| */
    // ros::ServiceClient deep_learning_client = nh.serviceClient<race_deep_learning_feature_extraction::vgg16_model>("/vgg_service");
    // ros::ServiceClient deep_learning_server = nh.serviceClient<race_deep_learning_feature_extraction::deep_representation>("/xception_service");
    deep_learning_server = nh.serviceClient<race_deep_learning_feature_extraction::deep_representation>(deep_learning_architecture);

    pp.info(std::ostringstream().flush() <<"deep_learning based simulated user -> Hello World");


	/* _________________________________________________
	|                                                   |
	|   create a subscriber to get recognition feedback |
	|___________________________________________________| */
	
	// unsigned found = name.find_last_of("/\\");
	// std::string topic_name = name.substr(0,found) + "/tracking/recognition_result";
	// ros::Subscriber sub = nh.subscribe(topic_name, 10, evaluationFunction);

	/* ______________________________
	|                                |
	|         create a publisher     |
	|________________________________| */
	// 
	// std::string pcin_topic = name.substr(0,found) + "/pipeline_default/tracker/tracked_object_point_cloud";  
	// ros::Publisher pub = nh.advertise< race_perception_msgs::PCTOV> (pcin_topic, 1000);

	/* _______________________________
	|                                |
	|         Initialization         |
	|________________________________| */

	string instance_path= "";	
	int class_index = 1; // class index
	unsigned int instance_number = 1;
	cat_id = 1;  // it is a constant to create a key for each category <Cat_Name><Cat_ID>
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
	ros::Time begin_process = ros::Time::now(); 
	ros::Time start_time = ros::Time::now();
// 	
	/* ______________________________
	|                                |
	|        Introduce category      |
	|________________________________| */
	introduceNewCategoryDeepLearningUsingGOOD(  home_address, 
												class_index,track_id,instance_number2.at(class_index-1), 
												evaluation_file,
												adaptive_support_lenght,
												global_image_width,
												threshold,
												number_of_bins,
												deep_learning_server);
		
	number_of_instances += 3; // we use three instances to initialize a category
	number_of_taught_categories ++;
	category_introduced << "1\n";
	
	vector <ObjectCategory> list_of_object_category = _pdb->getAllObjectCat();

	/* ______________________________
	|                                |
	|        Simulated Teacher       |
	|________________________________| */
	
	float precision = 0;
	float recall = 0;
	float f1 =0;
	vector <float> average_class_precision;
	
	while ( class_index < number_of_categories)  // one category is already taught above
	{
	    class_index ++; // class index
	    instance_path = "";
      
	    if ( introduceNewCategoryDeepLearningUsingGOOD( home_address, 
														class_index,track_id, instance_number2.at(class_index-1),
														evaluation_file,
														adaptive_support_lenght,
														global_image_width,
														threshold,
														number_of_bins, 
														deep_learning_server) == -1)  
	    {
			ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");
			ros::Duration duration = ros::Time::now() - begin_process;
			
			reportCurrentResults( TP, FP, FN, evaluation_file, true);

			reportExperimentResult( average_class_precision,
									number_of_instances, 
									number_of_taught_categories,  
									evaluation_file, duration);
						
			reportAllExperimentalResults( TP, FP, FN, obj_num,			    
											average_class_precision,
											number_of_instances, 
											number_of_taught_categories,
											name_of_approach );

			category_introduced.close();
			
			monitorF1VsLearnedCategory( f1_vs_learned_category, TP, FP, FN);

			plotSimulatedTeacherProgressInMatlab( experiment_number, protocol_threshold, precision_file);
			//plotLocalF1VsNumberOfLearnedCategoriesInMatlab( experiment_number, protocol_threshold, local_f1_vs_learned_category.c_str());
			plotGlobalF1VsNumberOfLearnedCategoriesInMatlab( experiment_number, f1_vs_learned_category.c_str());
			plotNumberOfLearnedCategoriesVsIterationsInMatlab( experiment_number, category_introduced_txt.c_str());
			plotNumberOfStoredInstancesPerCategoryInMatlab( list_of_object_category);

			// system_command= "cp "+ home_address+ "/Category/Category.txt " + ros::package::getPath("rug_simulated_user") + "/result/experiment_1" ;
			// system( system_command.c_str());
			return (0) ;
	    }

	
	    number_of_instances += 3; // we use three instances to initialize a category
	    number_of_taught_categories ++;
	    category_introduced << "1\n";
	    tp_tmp = 0; fp_tmp = 0; fn_tmp = 0;
	    
	    int k = 0; // number of classification results
	    float precision_tmp = 0;
	    precision = 0;
	    f1 = 0;
	    recall = 0;
	    unsigned int c = 1; // class index
	    int iterations = 1;
	    int iterations_user_sees_no_improvment = 0;
	    bool user_sees_no_improvement = false; // In the current implementation, If the simulated teacher 
															// sees the precision doesn't improve in 100 iteration, then, 
															// it terminares evaluation of the system, originally, 
															// it was an empirical decision of the human instructor
	    while ( ((f1 < protocol_threshold ) or (k < number_of_taught_categories)) and (!user_sees_no_improvement) )
	    {
			category_introduced<< "0\n";
			ROS_INFO("\t\t[-] Iteration:%i",iterations);
			ROS_INFO("\t\t[-] c:%i",c);
			ROS_INFO("\t\t[-] Instance number:%i",instance_number2.at(c-1));
			//info for debug
			ROS_INFO("\t\t[-] Home address parameter : %s", home_address.c_str());
			ROS_INFO("\t\t[-] number_of_categories : %i", number_of_categories);
			ROS_INFO("\t\t[-] protocol_threshold : %lf", protocol_threshold);
			ROS_INFO("\t\t[-] user_sees_no_improvment_const : %i", user_sees_no_improvment_const);
			ROS_INFO("\t\t[-] window_size : %i", window_size);

			// select an instance from an specific category
			instance_path= "";
			selectAnInstancefromSpecificCategory(c, instance_number2.at(c-1), instance_path);
			ROS_INFO("\t\t[-]-Test Instance: %s", instance_path.c_str());
		
			// check the selected instance exist or not? 
			if (instance_path.size() < 2) 
			{
				ROS_INFO("\t\t[-]-The %s file does not exist", instance_path.c_str());
				ROS_INFO("\t\t[-]- number of taught categories= %i", number_of_taught_categories); 
				ROS_INFO("Note: the experiment is terminated because there is not enough test data to continue the evaluation");
				category_introduced.close();

				ros::Duration duration = ros::Time::now() - begin_process;
			
				reportExperimentResult( average_class_precision,
										  number_of_instances, 
										  number_of_taught_categories,  
										  evaluation_file, duration);		    		    
				
				reportCurrentResults( TP, FP, FN, evaluation_file, true);
	
				reportAllExperimentalResults( TP, FP, FN, obj_num,			    
											  average_class_precision,
											  number_of_instances, 
											  number_of_taught_categories,
											  name_of_approach);
		
				monitorF1VsLearnedCategory( f1_vs_learned_category, TP, FP, FN);
				
				plotSimulatedTeacherProgressInMatlab( experiment_number, protocol_threshold, precision_file);
				// plotLocalF1VsNumberOfLearnedCategoriesInMatlab( experiment_number, protocol_threshold, local_f1_vs_learned_category.c_str());
				plotGlobalF1VsNumberOfLearnedCategoriesInMatlab( experiment_number, f1_vs_learned_category.c_str());
				plotNumberOfLearnedCategoriesVsIterationsInMatlab( experiment_number, category_introduced_txt.c_str());
				plotNumberOfStoredInstancesPerCategoryInMatlab( list_of_object_category);


				// system_command = "cp " + home_address + "/Category/Category.txt " + ros::package::getPath("rug_simulated_user") + "/result/experiment_1";
				// system( system_command.c_str());
				return (0) ;	    
			}
			else
			{
				std::string ground_truth_category_name = extractCategoryName(instance_path);
				instance_path = home_address + "/" + instance_path.c_str();
				
				//load an instance from file
				boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
				if (io::loadPCDFile <PointXYZRGBA> (instance_path.c_str(), *target_pc) == -1)
				{	
					ROS_ERROR("\t\t[-]-Could not read given object %s :", instance_path.c_str());
					return(0);
				}		   
				ROS_INFO("\t\t[-]-  track_id: %i , \tview_id: %i ", track_id, view_id );
				
					//// downsampling 
					// boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
					// pcl::VoxelGrid<PointT > voxelized_point_cloud;	
					// voxelized_point_cloud.setInputCloud (PCDFile);
					// voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
					// voxelized_point_cloud.filter (*target_pc);
				
				/* ___________________________
				|            	 			  |
				|    Object Representation    |
				|_____________________________| */

					////Declare PCTOV msg 
					// boost::shared_ptr<race_perception_msgs::PCTOV> msg (new race_perception_msgs::PCTOV );
					// pcl::toROSMsg(*cloud_filtered, msg->point_cloud);
					// msg->track_id = track_id;//it is 
					// msg->view_id = view_id;		    
					// msg->ground_truth_name = instance_path;//extractCategoryName(instance_path);
					// pub.publish (msg);
					// ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", instance_path.c_str());
						
					// start_time = ros::Time::now();
					// while (ros::ok() && (ros::Time::now() - start_time).toSec() < 2)
					// { /*wait*/}
					// ros::spinOnce();	


				/* _________________________________
				|                                   |
				|  option1: GOOD shape description  |
				|___________________________________| */

				boost::shared_ptr<pcl::PointCloud<PointT> > pca_object_view (new PointCloud<PointT>);
				boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
				vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
				double largest_side = 0;
				int sign = 1;
				vector <float> view_point_entropy;
				string std_name_of_sorted_projected_plane;
				Eigen::Vector3f center_of_bbox;
				vector< float > object_description;
				
				compuet_object_description( target_pc,
				                            adaptive_support_lenght,
				                            global_image_width,
				                            threshold,
				                            number_of_bins,
				                            pca_object_view,
				                            center_of_bbox,
				                            vector_of_projected_views, 
				                            largest_side, 
				                            sign,
				                            view_point_entropy,
				                            std_name_of_sorted_projected_plane,
				                            object_description );
				
				SITOV object_representation;
				for (size_t i = 0; i < object_description.size(); i++)
				{
				    object_representation.spin_image.push_back(object_description.at(i));
				}

				SITOV deep_object_representation;
				/// call deep learning service to represent the given GOOD description as vgg16  
				race_deep_learning_feature_extraction::deep_representation srv;
				srv.request.good_representation = object_representation.spin_image;
				if (deep_learning_server.call(srv))
				{
					for (size_t i = 0; i < srv.response.deep_representation.size(); i++)
					{
						deep_object_representation.spin_image.push_back(srv.response.deep_representation.at(i));
					}
				}
				else
				{
					ROS_ERROR("Failed to call deep learning service ");
				}

				ROS_INFO("\t\t[-] Size of deep object representation is %d", deep_object_representation.spin_image.size());


				/* ______________________
				|                        |
				|   Object Recognition   |
				|________________________| */
			
				//// get list of all object categories
				list_of_object_category = _pdb->getAllObjectCat();
				ROS_INFO(" %d categories exist in the perception database ", list_of_object_category.size() );
				
				ros::Time start_time_recognition = ros::Time::now();

				vector <float> object_category_distance;            
				for (size_t i = 0; i < list_of_object_category.size(); i++) // retrieves all categories from perceptual memory
				{
					if (list_of_object_category.at(i).rtov_keys.size() > 1) // check the size of category
					{    
						
						//ROS_INFO( "%s category has %d views",  list_of_object_category.at(i).cat_name.c_str(), list_of_object_category.at(i).rtov_keys.size());

						/* _____________________________________________
						|                                              |
						|   retrieves all instacnes of each category   |
						|______________________________________________| */
						
						std::vector<SITOV> category_instances; 
						for (size_t j = 0; j < list_of_object_category.at(i).rtov_keys.size(); j++) 
						{
							vector<SITOV> objectViewHistogram = _pdb->getSITOVs(list_of_object_category.at(i).rtov_keys.at(j).c_str());

							if (objectViewHistogram.size() > 0)
								category_instances.push_back(objectViewHistogram.at(0));                      
						}
								
						/* ________________________________________________________________________________________
						|                                                                                          |
						|   Compute the dissimilarity between the target object and all instances of a category    |
						|__________________________________________________________________________________________| */
						
						float min_distance_object_category;
						int best_matched_index;
						float normalized_distance;

						
						if (distance_function == "euclidean") 
						{                    
							// euclidean distance
							euclideanBasedObjectCategoryDistance( deep_representation_sitov, category_instances, min_distance_object_category, best_matched_index, pp);
						}
						else if (distance_function == "chi-sq") 
						{                    
							// chi-squared distance
							chiSquaredBasedObjectCategoryDistance( deep_representation_sitov, category_instances, min_distance_object_category, best_matched_index, pp);
						}
						else if (distance_function == "klbased") 
						{                    
							// symmetric Kullback–Leibler divergence 
							kLBasedObjectCategoryDistance( deep_representation_sitov, category_instances, min_distance_object_category, best_matched_index, pp);
						}
						else if (distance_function == "fidelity") 
						{                    
							// fidelity
							FidelityBasedObjectCategoryDistance( deep_representation_sitov, category_instances, min_distance_object_category, best_matched_index, pp); //added A&K
						}
						else if (distance_function == "sq-chord") 
						{                    
							// squared chord distance
							squaredChordBasedObjectCategoryDistance( deep_representation_sitov, category_instances, min_distance_object_category, best_matched_index, pp);
						} else {
							printf("Unknown distance function, exiting program.\n");
							return 1;
						}
									
						object_category_distance.push_back(min_distance_object_category);
					}
					else 
					{
						object_category_distance.push_back(1000000000);
					}
				}

				//// Timer toc
				ros::Duration duration = ros::Time::now() - begin_process;
				double duration_sec = duration.toSec();

				//// Object Recognition 
				begin_process = ros::Time::now();
				float sigma_distance = 0;
				float recognition_threshold = 1000000000;
				int categoryIndex = -1; 
				float minimum_distance = 9000000000;

				////simple nearest neighbor classifier
				findClosestCategory(  object_category_distance, 
										categoryIndex, 
										minimum_distance, 
										pp, 
										sigma_distance);

				std::string result_string;
				if (categoryIndex == -1)
				{
					ROS_INFO("Predicted category is unknown");
					result_string = "unknown";
				}
				else
				{
					ROS_INFO("Predicted category is %s", list_of_object_category.at(categoryIndex).cat_name.c_str());
					result_string = list_of_object_category.at(categoryIndex).cat_name.c_str();
				}
				
				ROS_INFO("****========***** Object recognition process took = %f", (ros::Time::now() - start_time_recognition).toSec());

				//// RRTOV stands for Recognition Result of Track Object View - it is a msg defined in race_perception_msgs
				RRTOV rrtov;
				rrtov.header.stamp = ros::Time::now();
				rrtov.track_id = track_id;
				rrtov.view_id = view_id;
				rrtov.recognition_result = result_string;
				rrtov.minimum_distance = minimum_distance;
				rrtov.ground_truth_name = instance_path.c_str();
				
				evaluationFunction(rrtov);
			
				// start_time = ros::Time::now();
				// while (ros::ok() && (ros::Time::now() - start_time).toSec() < 0.2)
				// { //wait
				// }
				// ros::spinOnce();


				if (c >= number_of_taught_categories)
				{
					c = 1;
				}
				else
				{
					c++;
				}

				//   boost::shared_ptr<race_perception_msgs::PCTOV> msg (new race_perception_msgs::PCTOV );
				//   pcl::toROSMsg(*PCDFile, msg->point_cloud);
				//   msg->track_id = track_id;//it is 
				//   msg->view_id = view_id;		    
				//   msg->ground_truth_name = instance_path;//extractCategoryName(instance_path);
				//   pub.publish (msg);
				//ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", instance_path.c_str());
			    


				if ( (iterations >= number_of_taught_categories) and (iterations <= window_size * number_of_taught_categories))
				{    
					if (( tp_tmp + fp_tmp) != 0)
					{
						precision = tp_tmp / double (tp_tmp+fp_tmp);
					}
					else
					{
						precision = 0;
					}		
					if ((tp_tmp + fn_tmp) != 0)
					{
						recall = tp_tmp / double (tp_tmp+fn_tmp);
					}
					else
					{
						recall = 0;
					}
					if ((precision + recall) != 0)
					{
						f1 = 2 * (precision * recall) / (precision + recall);
					}
					else
					{
						f1 = 0;
					}
						
					monitorPrecision (precision_file, f1);		

					if (f1 > protocol_threshold)
					{
						average_class_precision.push_back(f1);
						user_sees_no_improvement = false;
						ROS_INFO("\t\t[-]- precision= %f", precision);
						ROS_INFO("\t\t[-]- f1 = %f", f1); 
						reportCurrentResults(tp_tmp, fp_tmp, fn_tmp, evaluation_file, false);
						iterations = 1;
						monitorPrecision (local_f1_vs_learned_category, f1);
						ros::spinOnce();		
					}  
						
				}//if
				else if ( (iterations > window_size * number_of_taught_categories)) // In this condition, if we are at iteration I>3n, we only
																					// compute precision as the average of last 3n, and discart the first
																					// I-3n iterations.
				{
					//compute f1 of last 3n, and discart the first I-3n iterations
					f1 = computeF1OfLast3n (recognition_results, number_of_taught_categories);
					// ROS_INFO("\t\t[-]- precision= %f", precision);
					monitorPrecision (precision_file, f1);		
					reportF1OfLast3n (evaluation_file, f1);

					if (f1 > protocol_threshold)  
					{
						average_class_precision.push_back(f1);
						user_sees_no_improvement = false;
						reportCurrentResults(tp_tmp, fp_tmp, fn_tmp, evaluation_file, false);
						monitorPrecision (local_f1_vs_learned_category, f1);
						iterations = 1;
						iterations_user_sees_no_improvment = 0;
						ros::spinOnce();		
					} 
					else 
					{
						iterations_user_sees_no_improvment ++;
						ROS_INFO("\t\t[-]- %i user_sees_no_improvement_in_f1", iterations_user_sees_no_improvment);

						if (iterations_user_sees_no_improvment > user_sees_no_improvment_const)
						{
							average_class_precision.push_back(f1);

							user_sees_no_improvement = true;
							ROS_INFO("\t\t[-]- user_sees_no_improvement");
							ROS_INFO("\t\t[-]- Finish"); 
							ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 
							
							summary_of_experiment.open (evaluation_file.c_str(), std::ofstream::app);
							summary_of_experiment << "\n After " << user_sees_no_improvment_const <<" iterations, user sees no improvement in precision";
							summary_of_experiment.close();

							monitorPrecision( local_f1_vs_learned_category, f1);
							monitorF1VsLearnedCategory( f1_vs_learned_category, TP, FP, FN );

							
							ros::Duration duration = ros::Time::now() - begin_process;
							reportExperimentResult( average_class_precision,
													  number_of_instances, 
													  number_of_taught_categories,  
													  evaluation_file, duration);
							category_introduced.close();

							reportCurrentResults( TP, FP, FN, evaluation_file, true);
							
							reportAllExperimentalResults( TP, FP, FN, obj_num,			    
															average_class_precision,
															number_of_instances, 
															number_of_taught_categories,
															name_of_approach);
							
							plotSimulatedTeacherProgressInMatlab( experiment_number, protocol_threshold, precision_file);
							// plotLocalF1VsNumberOfLearnedCategoriesInMatlab( experiment_number, protocol_threshold, local_f1_vs_learned_category.c_str());
							plotGlobalF1VsNumberOfLearnedCategoriesInMatlab( experiment_number, f1_vs_learned_category.c_str());
							plotNumberOfLearnedCategoriesVsIterationsInMatlab( experiment_number, category_introduced_txt.c_str());
							plotNumberOfStoredInstancesPerCategoryInMatlab( list_of_object_category);

							// system_command = "cp " + home_address+"/Category/Category.txt " + ros::package::getPath("rug_simulated_user")+ "/result/experiment_1" ;
							// ROS_INFO("\t\t[-]- %s", system_command.c_str()); 
							// system( system_command.c_str());
							
							
							return 0 ;
						}
					}
				}
				else
				{
					float f1_system =0;
	
					if ((tp_tmp + fp_tmp)!=0)
					{
						precision = tp_tmp / double(tp_tmp+fp_tmp);
					}
					else
					{
						precision = 0;
					}		
					if ((tp_tmp+fn_tmp)!=0)
					{
						recall = tp_tmp / double(tp_tmp+fn_tmp);
					}
					else
					{
						recall = 0;
					}
					if ((precision + recall) != 0)
					{
						f1_system = 2 * (precision * recall) / (precision + recall);
					}
					else
					{
						f1_system = 0;
					}
					
					monitorPrecision( precision_file, f1_system);
				}
				k++; // k<-k+1 : number of classification result
				iterations	++;
			}//else	 
	    }//while
		monitorF1VsLearnedCategory (f1_vs_learned_category, TP, FP, FN );
	}
	
	ROS_INFO("\t\t[-]- Finish"); 
	ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 

	//get toc
	ros::Duration duration = ros::Time::now() - begin_process;
	
	monitorPrecision( local_f1_vs_learned_category, f1);
	monitorF1VsLearnedCategory( f1_vs_learned_category, TP, FP, FN );

	reportCurrentResults( TP, FP, FN, evaluation_file, true);
	reportExperimentResult( average_class_precision,
							  number_of_instances, 
							  number_of_taught_categories,  
							  evaluation_file, duration);	
	
	category_introduced.close();
	reportAllExperimentalResults( TP, FP, FN, obj_num,			    
									average_class_precision,
									number_of_instances, 
									number_of_taught_categories,
									name_of_approach);
	
	plotSimulatedTeacherProgressInMatlab(experiment_number, protocol_threshold, precision_file);
	plotLocalF1VsNumberOfLearnedCategoriesInMatlab (experiment_number, protocol_threshold, local_f1_vs_learned_category.c_str());
	plotGlobalF1VsNumberOfLearnedCategoriesInMatlab (experiment_number, f1_vs_learned_category.c_str());
	plotNumberOfLearnedCategoriesVsIterationsInMatlab (experiment_number, category_introduced_txt.c_str());
	plotNumberOfStoredInstancesPerCategoryInMatlab( list_of_object_category);

	// system_command= "cp " + home_address+"/Category/Category.txt " + ros::package::getPath("rug_simulated_user")+ "/result/experiment_1" ;
	// ROS_INFO("\t\t[-]- %s", system_command.c_str()); 
	// system( system_command.c_str());
		    
    return 0 ;
}






// 	    while ( ((f1 < protocol_threshold ) or (k < number_of_taught_categories)) and (!user_sees_no_improvement) )
// 	    {
// 			category_introduced<< "0\n";
// 			ROS_INFO("\t\t[-] Iteration:%i",iterations);
// 			ROS_INFO("\t\t[-] c:%i",c);
// 			ROS_INFO("\t\t[-] Instance number:%i", instance_number2.at(c-1));
// 			//info for debug
// 			ROS_INFO("\t\t[-] Home address parameter : %s", home_address.c_str());
// 			ROS_INFO("\t\t[-] number_of_categories : %i", number_of_categories);
// 			ROS_INFO("\t\t[-] protocol_threshold : %lf", protocol_threshold);
// 			ROS_INFO("\t\t[-] user_sees_no_improvment_const : %i", user_sees_no_improvment_const);
// 			ROS_INFO("\t\t[-] window_size : %i", window_size);

// 			// select an instance from an specific category
// 			InstancePath= "";
// 			selectAnInstancefromSpecificCategory(c, instance_number2.at(c-1), InstancePath);
// 			ROS_INFO("\t\t[-] Test Instance: %s", InstancePath.c_str());
		
// 			// check the selected instance exist or not? if yes, send it to the race_feature_extractor
// 			if (InstancePath.size() < 2) 
// 			{
// 				ROS_INFO("\t\t[-] The %s file does not exist", InstancePath.c_str());
// 				category_introduced.close();
// 				ROS_INFO("\t\t[-] number of taught categories= %i", number_of_taught_categories); 
// 				ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");

// 				ros::Duration duration = ros::Time::now() - beginProc;
			
// 				report_experiment_result ( average_class_precision,
// 											number_of_instances, 
// 											number_of_taught_categories,  
// 											evaluationFile, duration);		    		    
				
// 				report_current_results(TP,FP,FN,evaluationFile,true);
	
// 				report_all_experiments_results (TP, FP, FN, Obj_Num,			    
// 												average_class_precision,
// 												number_of_instances, 
// 												number_of_taught_categories,
// 												name_of_approach );
		
// 				monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );
				
// 				Visualize_simulated_teacher_in_MATLAB(RunCount, protocol_threshold, precision_file);
// 				Visualize_Local_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, protocol_threshold, local_F1_vs_learned_category.c_str());
// 				Visualize_Global_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, F1_vs_learned_category.c_str());
// 				Visualize_Number_of_learned_categories_vs_Iterations (RunCount, category_introduced_txt.c_str());

// 				systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
// 				system( systemStringCommand.c_str());
// 				return (0) ;	    
// 			}
// 			else
// 			{
// 				std::string ground_truth_category_name =extractCategoryName(InstancePath);
// 				InstancePath = home_address +"/"+ InstancePath.c_str();
				
// 				//load an instance from file
// 				boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
// 				if (io::loadPCDFile <PointXYZRGBA> (InstancePath.c_str(), *target_pc) == -1)
// 				{	
// 					ROS_ERROR("\t\t[-] Could not read given object %s :",InstancePath.c_str());
// 					return(0);
// 				}		   
// 				ROS_INFO("\t\t[-] track_id: %i , \tview_id: %i ",track_id, view_id );
				
// 				// boost::shared_ptr<PointCloud<PointT> > cloud_filtered (new PointCloud<PointT>);
// 				// pcl::VoxelGrid<PointT > voxelized_point_cloud;	
// 				// voxelized_point_cloud.setInputCloud (target_pc);
// 				// voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
// 				// voxelized_point_cloud.filter (*cloud_filtered);
				
// 				// //Declare PCTOV msg 
// 				// boost::shared_ptr<race_perception_msgs::PCTOV> msg (new race_perception_msgs::PCTOV );
// 				// pcl::toROSMsg(*cloud_filtered, msg->point_cloud);
// 				// msg->track_id = track_id;//it is 
// 				// msg->view_id = view_id;		    
// 				// msg->ground_truth_name = InstancePath;//extractCategoryName(InstancePath);
// 				// pub.publish (msg);
// 				// ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", InstancePath.c_str());
			






// 				/* __________________________________________________________
// 				|                                                           |
// 				|  Compute the new shape description for given point cloud  |
// 				|___________________________________________________________| */


// 				ros::Time begin_Proc = ros::Time::now(); //start tic	

// 				boost::shared_ptr<pcl::PointCloud<PointT> > pca_object_view (new PointCloud<PointT>);
// 				boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
// 				vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
// 				double largest_side = 0;
// 				int  sign = 1;
// 				vector <float> view_point_entropy;
// 				string std_name_of_sorted_projected_plane;
// 				Eigen::Vector3f center_of_bbox;
// 				vector< float > object_description;
				
// 				compuet_object_description( target_pc,
// 											adaptive_support_lenght,
// 											global_image_width,
// 											threshold,
// 											number_of_bins,
// 											pca_object_view,
// 											center_of_bbox,
// 											vector_of_projected_views, 
// 											largest_side, 
// 											sign,
// 											view_point_entropy,
// 											std_name_of_sorted_projected_plane,
// 											object_description );
				
				
// 				SITOV object_representation;
// 				for (size_t i = 0; i < object_description.size(); i++)
// 				{
// 					object_representation.spin_image.push_back(object_description.at(i));
// 				}
// 				//ROS_INFO("\nsize of object view histogram %ld",object_representation.spin_image.size());
				
// 				SITOV deep_representation_sitov;
// 				/// call deep learning service to represent the given GOOD description as vgg16  
// 				race_deep_learning_feature_extraction::deep_representation srv;
// 				srv.request.good_representation = object_representation.spin_image;
// 				if (deep_learning_server.call(srv))
// 				{
// 					//pp.info(std::ostringstream().flush() << "################ receive server responce with size of " << srv.response.deep_representation.size() );
// 					for (size_t i = 0; i < srv.response.deep_representation.size(); i++)
// 					{
// 						deep_representation_sitov.spin_image.push_back(srv.response.deep_representation.at(i));
// 					}
// 				}
// 				else
// 				{
// 					ROS_ERROR("Failed to call deep learning service ");
// 				}


// 				ROS_INFO("\t\t[-] Size of deep object representation is %d", deep_representation_sitov.spin_image.size());

// 				boost::shared_ptr< vector <SITOV> > msg_out;
// 				msg_out = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
// 				// msg_out->push_back (object_representation);
// 				msg_out->push_back (deep_representation_sitov);
				
// 				//get toc
// 				ros::Duration duration = ros::Time::now() - begin_Proc;
// 				double duration_sec = duration.toSec();
// 				pp.info(std::ostringstream().flush() << "Compute description for the given point cloud took " << duration_sec << " secs");

// 				/* __________________________________________________________
// 				|                       	                                  |
// 				|  Write GOOD features to _crtov and Publish for recognition |
// 				|____________________________________________________________| */

				
// 				// //Declare SITOV (Spin Images of Tracked Object View)
// 				// SITOV _sitov;	    
				
// 				// //declare the RTOV complete variable
// 				// race_perception_msgs::CompleteRTOV _crtov;
// 				// _crtov.track_id = track_id;
// 				// _crtov.view_id = view_id;
// 				// _crtov.ground_truth_name = PCDFileAddress;
				
// 				// //Add the Spin Images in msg_out to Publish
// 				// for (size_t i = 0; i < msg_out->size(); i++)
// 				// {
// 				// _sitov = msg_out->at(i); //copy VFH
// 				// _sitov.track_id = track_id; //copy track_id
// 				// _sitov.view_id = view_id; //copy view_id
// 				// _sitov.spin_img_id = i; //copy spin image id

// 				// //Addd sitov to completertov sitov list
// 				// _crtov.sitov.push_back(_sitov);
// 				// }

// 				// //Publish the CompleteRTOV to recognition
// 				// //_p_crtov_publisher->publish (_crtov);







// 				///get list of all object categories
// 				vector <ObjectCategory> ListOfObjectCategory = _pdb->getAllObjectCat();
// 				vector <ObjectCategory> v_oc = _pdb->getAllObjectCat();
// 				vector <NOCD> normalizedObjectCategoriesDistanceMsg;
// 				vector <float> normalizedObjectCategoriesDistance;
// 				pp.info(std::ostringstream().flush() << ListOfObjectCategory.size()<<" categories exist in the perception database" );
// 				//ROS_INFO(" %d categories exist in the perception database ", ListOfObjectCategory.size() );

				
// 				for (size_t i = 0; i < ListOfObjectCategory.size(); i++) // all category exist in the database 
// 				{
// 					if (ListOfObjectCategory.at(i).rtov_keys.size() > 1) 
// 					{    
// 						pp.info(std::ostringstream().flush() << ListOfObjectCategory.at(i).cat_name.c_str() <<" category has " 
// 										<< ListOfObjectCategory.at(i).rtov_keys.size()<< " views");
// 						//ROS_INFO( "%s category has %d views",  ListOfObjectCategory.at(i).cat_name.c_str(), ListOfObjectCategory.at(i).rtov_keys.size());

// 						/// recognition    						
// 						///compute objectCategoryDistance() -> (return minimum distance and best_matched_index) 
// 						std::vector< SITOV > category_instances; 
// 						for (size_t j = 0; j < ListOfObjectCategory.at(i).rtov_keys.size(); j++)
// 						{
// 							vector< SITOV > objectViewHistogram = _pdb->getSITOVs(v_oc.at(i).rtov_keys.at(j).c_str());
							
// 							if (objectViewHistogram.size() > 0)
// 								category_instances.push_back(objectViewHistogram.at(0));

// 							//pp.info(std::ostringstream().flush() << "size of object view histogram = " << objectViewHistogram.size());
// 							//pp.info(std::ostringstream().flush() << "key for the object view histogram = " << v_oc.at(i).rtov_keys.at(j).c_str());
// 						}
// 						pp.info(std::ostringstream().flush() << "number of object views histogram retrived from database = " << category_instances.size());
					
// 						///Compute the absolute distance from collected view and all category view instances
// 						float ObjectCategoryDistanec;
// 						int best_matched_index;
// 						float normalizedDistance;

// 						///euclidean distance
// 						//histogramBasedObjectCategoryDistance(deep_representation_sitov,category_instances,ObjectCategoryDistanec,best_matched_index, pp);
						
// 						/// chi-squared distance
// 						chiSquaredBasedObjectCategoryDistance(deep_representation_sitov,category_instances,ObjectCategoryDistanec,best_matched_index, pp);
						
// 						///KL distance
// 						//histogramBasedObjectCategoryKLDistance( deep_representation_sitov,category_instances,ObjectCategoryDistanec,best_matched_index, pp);

// 						///Nearest neighbor classification
// 						normalizedObjectCategoryDistance(ObjectCategoryDistanec,1,normalizedDistance, pp);
									
// 						normalizedObjectCategoriesDistance.push_back(normalizedDistance);

// 						NOCD NOCDtmp;
// 						std::string oc_key = _pdb->makeOCKey(key::OC, ListOfObjectCategory.at(i).cat_name.c_str(), ListOfObjectCategory.at(i).cat_id);
// 						NOCDtmp.object_category_key= oc_key.c_str();
// 						NOCDtmp.normalized_distance=normalizedDistance;
// 						//NOCDtmp.best_matched_rtov_key= ListOfObjectCategory.at(i).rtov_keys.at(best_matched_index).c_str();
// 						NOCDtmp.cat_name = ListOfObjectCategory.at(i).cat_name;
// 						normalizedObjectCategoriesDistanceMsg.push_back(NOCDtmp);
// 					}
// 					else 
// 					{
// 						normalizedObjectCategoriesDistance.push_back(1000000);

// 						NOCD NOCDtmp;
// 						std::string oc_key = _pdb->makeOCKey(key::OC, ListOfObjectCategory.at(i).cat_name.c_str(), ListOfObjectCategory.at(i).cat_id);
// 						NOCDtmp.object_category_key= oc_key.c_str();
// 						NOCDtmp.normalized_distance=1000000;
// 						//NOCDtmp.best_matched_rtov_key= ListOfObjectCategory.at(i).rtov_keys.at(best_matched_index).c_str();
// 						NOCDtmp.cat_name = ListOfObjectCategory.at(i).cat_name;
// 						normalizedObjectCategoriesDistanceMsg.push_back(NOCDtmp);

// 						//pp.warn(std::ostringstream().flush() << "There is no data about object views in " <<ListOfObjectCategory.at(i).cat_name.c_str() << " category");
// 						pp.warn(std::ostringstream().flush() << "category size should larger than two, otherwise, we can not compute the ICD, ICD= 0.001");
// 					}
// 				}

// 				///Print normalized distance from collected view to all category instances
// 				for (size_t k = 0; k < normalizedObjectCategoriesDistanceMsg.size(); k++)
// 				{
// 					pp.info(std::ostringstream().flush() << "NormalizedDistance (target_object, " 
// 						<< normalizedObjectCategoriesDistanceMsg.at(k).cat_name.c_str()<< " category) = "
// 						<< normalizedObjectCategoriesDistance.at(k));
// 				}

// 				//Timer toc
// 				duration = ros::Time::now() - beginProc;
// 				duration_sec = duration.toSec();
// 				pp.info(std::ostringstream().flush() << "Compute ND (O, Ci) took " << duration_sec << " secs");

			
// 				/// Classification Rule and Confidence value computation 
// 				beginProc = ros::Time::now();
// 				double sigma_distance = 0;
// 				double recognition_threshold = 1000000000000000;
// 				/// Sorting the normalizedObjectCategoriesDistanceMsg vector
// 				for (size_t i = 0; i < normalizedObjectCategoriesDistanceMsg.size(); i++)
// 				{
// 					for (size_t j = i; j < normalizedObjectCategoriesDistanceMsg.size(); j++)
// 					{	
// 						if (normalizedObjectCategoriesDistanceMsg.at(i).normalized_distance > normalizedObjectCategoriesDistanceMsg.at(j).normalized_distance )
// 						{
// 							NOCD NOCDtmp;
// 							NOCDtmp=normalizedObjectCategoriesDistanceMsg.at(i);
// 							normalizedObjectCategoriesDistanceMsg.at(i)=normalizedObjectCategoriesDistanceMsg.at(j);
// 							normalizedObjectCategoriesDistanceMsg.at(j)=NOCDtmp;
// 						}
// 					}
// 				}

// 				//confidenceValue = 1 - (minimumDistance/sigma_distance);
// 				int categoryIndex = -1; //Added by mike (=-1) because of the if after and the function returns withount giving a value to categoryindex
// 				float confidenceValue = 0;
// 				float minimum_distance = 100000000;
// 				simpleClassificationRule(normalizedObjectCategoriesDistance, 
// 										categoryIndex, 
// 										confidenceValue, 
// 										(float)recognition_threshold,
// 										minimum_distance,
// 										pp);	

// 				std::string result_string;
// 				if (categoryIndex == -1)
// 				{
// 					pp.info(std::ostringstream().flush() << "Predicted category is unknown" );
// 					result_string = "Unknown";
// 				}
// 				else
// 				{
// 					pp.info(std::ostringstream().flush() << "Predicted category is "<<  ListOfObjectCategory.at(categoryIndex).cat_name.c_str());
// 					pp.info(std::ostringstream().flush() << "Confidence value = "<<  confidenceValue);
// 					result_string = ListOfObjectCategory.at(categoryIndex).cat_name.c_str();
// 				}
				
// 				RRTOV _rrtov;
// 				_rrtov.header.stamp = ros::Time::now();
// 				_rrtov.track_id = track_id;
// 				_rrtov.view_id = view_id;
// 				_rrtov.recognition_result = result_string;
// 				_rrtov.minimum_distance = minimum_distance;
// 				_rrtov.ground_truth_name = InstancePath.c_str();
// 				_rrtov.result = normalizedObjectCategoriesDistanceMsg;
				
// 				//pp.printCallback();

// 				evaluationfunction(_rrtov);


// 				ros::spinOnce();	
		    
// 				if (c >= number_of_taught_categories)
// 				{
// 					c = 1;
// 				}
// 				else
// 				{
// 					c++;
// 				}


// 				//ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", InstancePath.c_str());

// 				if ( (iterations >= number_of_taught_categories) and (iterations <= window_size * number_of_taught_categories))
// 				{    
// 					if ((TPtmp+FPtmp)!=0)
// 					{
// 						Precision = TPtmp/double (TPtmp+FPtmp);
// 					}
// 					else
// 					{
// 						Precision = 0;
// 					}		
// 					if ((TPtmp+FNtmp)!=0)
// 					{
// 						Recall = TPtmp/double (TPtmp+FNtmp);
// 					}
// 					else
// 					{
// 						Recall = 0;
// 					}
// 					if ((Precision + Recall)!=0)
// 					{
// 						F1 = 2 * (Precision * Recall )/(Precision + Recall );
// 					}
// 					else
// 					{
// 						F1 = 0;
// 					}
			    
// 					monitor_precision (precision_file, F1);		

// 					if (F1 > protocol_threshold)
// 					{
// 						average_class_precision.push_back(F1);
// 						User_sees_no_improvement_in_precision = false;
// 						ROS_INFO("\t\t[-]- Precision= %f", Precision);
// 						ROS_INFO("\t\t[-]- F1 = %f", F1); 
// 						report_current_results(TPtmp,FPtmp,FNtmp,evaluationFile,false);
// 						iterations = 1;
// 						monitor_precision (local_F1_vs_learned_category, F1);
// 						ros::spinOnce();		
// 					}  
			    
// 				}//if
// 				else if ( (iterations > window_size * number_of_taught_categories)) // In this condition, if we are at iteration I>3n, we only
// 																							// compute precision as the average of last 3n, and discart the first
// 																							// I-3n iterations.
// 				{
// 					//compute precision of last 3n, and discart the first I-3n iterations
// 					F1 = compute_Fmeasure_of_last_3n (recognition_results, number_of_taught_categories);
// 					//ROS_INFO("\t\t[-]- Precision= %f", Precision);
// 					monitor_precision (precision_file, F1);		
// 					report_F1_of_last_3n (evaluationFile, F1);
				
// 					if (F1 > protocol_threshold)  
// 					{
// 						//average_class_precision.push_back(Precision);
// 						average_class_precision.push_back(F1);

// 						User_sees_no_improvement_in_precision = false;
// 						report_current_results(TPtmp,FPtmp,FNtmp,evaluationFile,false);
// 						monitor_precision (local_F1_vs_learned_category, F1);
// 						iterations = 1;
// 						iterations_user_sees_no_improvment=0;
// 						ros::spinOnce();		
// 					} 
// 					else 
// 					{
// 						iterations_user_sees_no_improvment++;
// 						ROS_INFO("\t\t[-]- %i user_sees_no_improvement_in_F1", iterations_user_sees_no_improvment);

// 						if (iterations_user_sees_no_improvment > user_sees_no_improvment_const)
// 						{
// 							average_class_precision.push_back(F1);

// 							User_sees_no_improvement_in_precision = true;
// 							ROS_INFO("\t\t[-]- User_sees_no_improvement_in_precision");
// 							ROS_INFO("\t\t[-]- Finish"); 
// 							ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 
							
// 							Result.open (evaluationFile.c_str(), std::ofstream::app);
// 							Result << "\n After " << user_sees_no_improvment_const <<" iterations, user sees no improvement in precision";
// 							Result.close();

// 							monitor_precision (local_F1_vs_learned_category, F1);
// 							monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );

							
// 							ros::Duration duration = ros::Time::now() - beginProc;
// 							report_experiment_result ( average_class_precision,
// 														number_of_instances, 
// 														number_of_taught_categories,  
// 														evaluationFile, duration);
// 							category_introduced.close();

// 							report_current_results(TP,FP,FN,evaluationFile,true);
							

// 							/// int total_number_of_experiments = 50;//TODO: shold be an input paramer for report_all_experiments_results
// 							report_all_experiments_results (TP, FP, FN, Obj_Num,			    
// 															average_class_precision,
// 															number_of_instances, 
// 															number_of_taught_categories,
// 															name_of_approach);
							
// 							Visualize_simulated_teacher_in_MATLAB(RunCount, protocol_threshold, precision_file);
// 							Visualize_Local_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, protocol_threshold, local_F1_vs_learned_category.c_str());
// 							Visualize_Global_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, F1_vs_learned_category.c_str());
// 							Visualize_Number_of_learned_categories_vs_Iterations (RunCount, category_introduced_txt.c_str());
							
// 							systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
// 							ROS_INFO("\t\t[-]- %s", systemStringCommand.c_str()); 
// 							system( systemStringCommand.c_str());
							
							
// 							return 0 ;
// 						}//if
// 					}//else 
// 		    	}//else if
// 				else
// 				{
						
// 					if ((TPtmp+FPtmp)!=0)
// 					{
// 						Precision = TPtmp/double (TPtmp+FPtmp);
// 					}
// 					else
// 					{
// 						Precision = 0;
// 					}		
// 					if ((TPtmp+FNtmp)!=0)
// 					{
// 						Recall = TPtmp/double (TPtmp+FNtmp);
// 					}
// 					else
// 					{
// 						Recall = 0;
// 					}
// 					if ((Precision + Recall)!=0)
// 					{
// 						F1System = 2 * (Precision * Recall )/(Precision + Recall );
// 					}
// 					else
// 					{
// 						F1System = 0;
// 					}
					
// 					monitor_precision (precision_file, F1System);
// 				}//else
// 				k++; // k<-k+1 : number of classification result
// 				iterations	++;
// 			}//else	 
// 		}//while F1 < threshold or user_sees_no_improvement_in_precision
// 		monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );
// 	}//while  class _index
	
// 	ROS_INFO("\t\t[-]- Finish"); 
// 	ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 

// 	//get toc
// 	ros::Duration duration = ros::Time::now() - beginProc;
	
// 	monitor_precision (local_F1_vs_learned_category, F1);
// 	monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );

// 	report_current_results(TP,FP,FN,evaluationFile,true);
// 	report_experiment_result (average_class_precision,
// 			    number_of_instances, 
// 			    number_of_taught_categories,  
// 			    evaluationFile, duration);	
	
// 	category_introduced.close();
// 	report_all_experiments_results (TP, FP, FN, Obj_Num,			    
// 				average_class_precision,
// 				number_of_instances, 
// 				number_of_taught_categories,
// 				name_of_approach);
	
// 	Visualize_simulated_teacher_in_MATLAB(RunCount, protocol_threshold, precision_file);
// 	Visualize_Local_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, protocol_threshold, local_F1_vs_learned_category.c_str());
// 	Visualize_Global_F1_vs_Number_of_learned_categories_in_MATLAB (RunCount, F1_vs_learned_category.c_str());
// 	Visualize_Number_of_learned_categories_vs_Iterations (RunCount, category_introduced_txt.c_str());
// 	systemStringCommand= "cp "+home_address+"/Category/Category.txt " + ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count ;
// 	ROS_INFO("\t\t[-]- %s", systemStringCommand.c_str()); 
// 	system( systemStringCommand.c_str());
// 	//RunCount++;
	    
//     return 0 ;
// }



