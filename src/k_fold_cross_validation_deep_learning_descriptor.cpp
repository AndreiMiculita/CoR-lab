// ############################################################################
//    
//   Created: 	2/07/2019
//   Author : 	Hamidreza Kasaei
//   Email  :	hamidreza.kasaei@rug.nl
//   Web    :   www.ai.rug.nl/hkasaei
//   Purpose:  This program provides a K-fold cross validation algorithm 
//             to evaluate an intasnce-based learning approach for 3D object 
//             recognition approaches in terms of precision and recall. 
//             In each iteration, a single fold is used for testing, and the 
//             remaining data are used as training data. The cross-validation 
//             process is then repeated 10 times, which each of the 10 folds 
//             used exactly once as the test data.  
// ############################################################################

/* _________________________________
   |                                 |
   |          RUN SYSTEM BY          |
   |_________________________________| */
   
//rm -rf /tmp/pdb
//roslaunch race_kfold_cross_validation_evaluation dictionary_based_K_fold_cross_validation.launch

/* _________________________________
  |                                 |
  |             INCLUDES            |
  |_________________________________| */
  
//system includes
#include <std_msgs/String.h>
#include <sstream>
#include <stdio.h>

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
#include <CGAL/Plane_3.h>
#include <pcl_conversions/pcl_conversions.h>

//package includes
#include <object_descriptor/object_descriptor_functionality.h>
#include <rug_kfold_cross_validation/rug_kfold_cross_validation_functionality.h>
#include <feature_extraction/spin_image.h>
#include <race_3d_object_tracking/TrackedObjectPointCloud.h>
#include <race_perception_msgs/perception_msgs.h>
#include <race_3d_object_tracking/TrackedObjectPointCloud.h>
#include <object_conceptualizer/object_conceptualization.h>
#include <race_perception_utils/print.h>
#include <race_perception_msgs/CompleteRTOV.h>
// 
#include <object_descriptor/object_descriptor_functionality.h>
#include <race_deep_learning_feature_extraction/deep_representation.h>

/* _________________________________
|                                 |
|            constant            |
|_________________________________| */
#define spin_image_width 8
#define subsample_spinimages 0
#define spin_image_support_lenght 0.05

/* _________________________________
   |                                 |
   |            NameSpace            |
   |_________________________________| */

using namespace std;
using namespace pcl;
using namespace ros;
using namespace race_perception_db;
using namespace race_perception_msgs;
using namespace race_perception_utils;

typedef pcl::PointXYZRGBA PointT;


PerceptionDB* _pdb;


/* __________________________
|                            |
|         Parameters         |
|____________________________| */

int  number_of_bins = 5;
int  adaptive_support_lenght = 1;
double global_image_width =0.5;
double downsampling_voxel_size = 0.001;
int threshold = 1000;	
string name_of_approach = "your_name+your_student_number+DL_model"; //CRC stands for Cognitive Robotics Course - can be a param

double recognition_threshold;

bool downsampling = false;
bool image_normalization = true;
bool multiviews = true;
bool max_pooling = true;
bool modelnet_dataset = false; // we do not need to perform PCA for modelNet dataset

std::string home_address= "/media/hamidreza/b9802402-9c38-4427-b601-84cff22d8e35/datasets/object_datasets/";

std::string deep_learning_architecture = "/vgg16_service";


/* _______________________________
|                                 |
|         Global Variable         |
|_________________________________| */

std::string evaluation_file;
ofstream results , train_csv, test_csv;
int TP =0, FP=0, FN=0, TPtmp =0, FPtmp=0, FNtmp=0, obj_no=0;


std::vector< std::vector <int> > confusion_matrix;
std::vector<string> map_category_name_to_index;


void evaluationFunction(const race_perception_msgs::RRTOV &result)
{

    PrettyPrint pp;
    string tmp = result.ground_truth_name.substr(home_address.size(),result.ground_truth_name.size());
    string true_category = extractCategoryName(tmp);
    std:: string object_name;
    object_name = extractObjectName (result.ground_truth_name);
    pp.info(std::ostringstream().flush() << "[-]object_name: "<< object_name.c_str()); 
    
    
    obj_no++;
    pp.info(std::ostringstream().flush() << "[-]track_id="<<result.track_id << "\tview_id=" << result.view_id);
    
    //// print the minimum distance of the given object to each category
    pp.info(std::ostringstream().flush() << "\n");
    pp.info(std::ostringstream().flush() << "[-]object_category_distance:");
    for (size_t i = 0; i < result.result.size(); i++)
    {
	    //pp.info(std::ostringstream().flush() << "-"<< result.result.at(i).normalized_distance);
        pp.info(std::ostringstream().flush() <<"\t D(target_object, "<<  result.result.at(i).cat_name.c_str() 
                                             <<") = "<< result.result.at(i).normalized_distance);
    }
    pp.info(std::ostringstream().flush() << "\n");
    
    if ( result.result.size() <= 0)
    {
	    pp.warn(std::ostringstream().flush() << "Warning: size of result is 0");
    }
       
    float minimum_distance = result.minimum_distance;

    string predict_category;
    predict_category= result.recognition_result.c_str();
    
    confusionMatrixGenerator(  true_category.c_str(), predict_category.c_str(), 
                               map_category_name_to_index,
                               confusion_matrix );

    pp.info(std::ostringstream().flush() << "[-]object_name: "<< object_name.c_str());
    pp.info(std::ostringstream().flush() << "[-]true_category: "<<true_category.c_str());
    pp.info(std::ostringstream().flush() << "[-]predict_category: " << predict_category.c_str());
    pp.info(std::ostringstream().flush() << "[-]minimum_distance: " << minimum_distance);


    char unknown[] = "unknown";
        
    results.open( evaluation_file.c_str(), std::ofstream::app);    
    results.precision(4);
    
    if ((strcmp(true_category.c_str(),unknown)==0) && (strcmp(predict_category.c_str(),unknown)==0))
    { 	
        results << "\n"<<obj_no<<"\t"<<object_name <<"\t\t\t"<< true_category <<"\t\t"<< predict_category <<"\t\t\t"<< "0\t0\t0"<< "\t\t"<< minimum_distance;
        results << "\n-----------------------------------------------------------------------------------------------------------------------------------";
    }
    else if ((strcmp(true_category.c_str(),predict_category.c_str())==0))
    { 
        TP++;
        TPtmp++;	
        results << "\n"<<obj_no<<"\t"<<object_name <<"\t\t\t"<< true_category <<"\t\t"<< predict_category <<"\t\t\t"<< "1\t0\t0" << "\t\t"<< minimum_distance;
        results << "\n-----------------------------------------------------------------------------------------------------------------------------------";
    }
    else if ((strcmp(true_category.c_str(),unknown)==0) && (strcmp(predict_category.c_str(),unknown)!=0))
    { 	
        FP++; 
        FPtmp++;
        results << "\n"<<obj_no<<"\t"<<object_name <<"\t\t\t"<< true_category <<"\t\t"<< predict_category <<"\t\t\t"<< "0\t1\t0"<< "\t\t"<< minimum_distance;
        results << "\n===================================================================================================================================";    }
    else if ((strcmp(true_category.c_str(),unknown)!=0) && (strcmp(predict_category.c_str(),unknown)==0))
    { 	
        FN++;
        FNtmp++;
        results << "\n"<<obj_no<<"\t"<<object_name <<"\t\t\t"<< true_category <<"\t\t"<< predict_category <<"\t\t\t"<< "0\t0\t1"<< "\t\t"<< minimum_distance;
        results << "\n===================================================================================================================================";
    }
    else if ((strcmp(true_category.c_str(),predict_category.c_str())!=0))
    {  	
        FP++; FN++;
        FPtmp++; FNtmp++;    
        results << "\n"<<obj_no<<"\t"<<object_name <<"\t\t\t"<< true_category <<"\t\t"<< predict_category <<"\t\t\t"<< "0\t1\t1"<< "\t\t"<< minimum_distance;
        results << "\n===================================================================================================================================";
    }
    results.close();
    pp.printCallback();
    ros::spinOnce();
    
}


int main(int argc, char** argv)
{
    ros::init (argc, argv, "EVALUATION");
    ros::NodeHandle nh;
    string name = nh.getNamespace();
    bool map_name_to_idx_flag = false;

    /* ______________________________
    |                                |
    |       initialize database      |
    |________________________________| */
    _pdb = race_perception_db::PerceptionDB::getPerceptionDB(&nh); //initialize the database class_list_macros

    PrettyPrint pp;
    ros::Time beginProc = ros::Time::now(); //start tic	

    //create a folder to save all relevant the resuts
    string systemStringCommand= "mkdir "+ ros::package::getPath("rug_kfold_cross_validation")+ "/result/experiment_1";
    system( systemStringCommand.c_str());
    
 
    /* ______________________________
    |                                |
    |         create a subscriber    |
    |________________________________| */

    ////create a subscriber to get recognition feedback
    // unsigned found = name.find_last_of("/\\");
    // std::string topic_name = name.substr(0,found) + "/tracking/recognition_result";
    // ros::Subscriber sub = nh.subscribe(topic_name, 10, evaluationfunction);
    

    /* _____________________________
    |                               |
    |         create a publisher    |
    |_______________________________| */

    // std::string pcin_topic = name.substr(0,found) + "/pipeline_default/tracker/tracked_object_point_cloud";  
    // ros::Publisher pub = nh.advertise< race_perception_msgs::PCTOV> (pcin_topic, 1);
        
    // boost::shared_ptr<ros::Publisher> _p_crtov_publisher;
    // _p_crtov_publisher = (boost::shared_ptr<ros::Publisher>) new ros::Publisher;
    // *_p_crtov_publisher = nh.advertise<race_perception_msgs::CompleteRTOV> ("/perception/pipeline_default/object_descriptor_deep_learning/new_histogram_tracked_object_view", 1000);
    
    /* _____________________________________
    |                                       |
    |    read prameters from launch file    |
    |_______________________________________| */

    // read database parameter
    nh.param<std::string>("/perception/home_address", home_address, home_address);
    
    // parameters of new object descriptor
    nh.param<int>("/perception/number_of_bins", number_of_bins, number_of_bins);
    nh.param<double>("/perception/global_image_width", global_image_width, global_image_width);		
    nh.param<int>("/perception/adaptive_support_lenght", adaptive_support_lenght, adaptive_support_lenght);
    nh.param<int>("/perception/threshold", threshold, threshold);
    
    // recognition threshold
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

    // read name_of_approach parameter
    nh.param<std::string>("/perception/name_of_approach", name_of_approach, name_of_approach);
    
    // read downsampling parameter 0 = FLASE, 1 = TRUE 
    nh.param<bool>("/perception/downsampling", downsampling, downsampling);
    string downsampling_flag = (downsampling == 0) ? "FALSE" : "TRUE";

    // read downsampling_voxel_size parameter
    nh.param<double>("/perception/downsampling_voxel_size", downsampling_voxel_size, downsampling_voxel_size);
    downsampling_voxel_size = downsampling_voxel_size * 0.02;

    // // read modelnet_dataset parameter 0 = FLASE, 1 = TRUE 
    // nh.param<bool>("/perception/modelnet_dataset", modelnet_dataset, modelnet_dataset);
    // string modelnet_dataset_flag = (modelnet_dataset == 0) ? "False" : "True";

    
    /* _____________________________
    |                               |
    |     create a client server    |
    |_______________________________| */
    // ros::ServiceClient deep_learning_client = nh.serviceClient<race_deep_learning_feature_extraction::vgg16_model>("/vgg_service");
    // ros::ServiceClient deep_learning_server = nh.serviceClient<race_deep_learning_feature_extraction::deep_representation>("/xception_service");
    ros::ServiceClient deep_learning_server = nh.serviceClient<race_deep_learning_feature_extraction::deep_representation>(deep_learning_architecture);

    pp.info(std::ostringstream().flush() <<"deep_learning based k-fold cross-validation -> Hello World");

    
    /* ______________________________________
    |                                       |
    |  Define path of train and test data   |
    |_______________________________________| */  
    
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string test_data_path = package_path + "/CV_test_instances.txt";
    string train_data_path = package_path + "/CV_train_instances.txt";


	int exp_num = 0;
	
    /* _____________________________________________________
    |							                            |  
    | expriments for varing Gussian Noise and downSampling  |
    |_______________________________________________________| */  

    // number_of_performed_experiments(name_of_approach, exp_num);
    string dataset= (home_address == "/home/cognitiverobotics/datasets/restaurant_object_dataset/") ? "Restaurant RGB-D Object Dataset" : "ModelNet10";

    //0 = FLASE, 1 = TRUE
    modelnet_dataset = (home_address == "/home/cognitiverobotics/datasets/restaurant_object_dataset/") ? false : true;
 
    evaluation_file = ros::package::getPath("rug_kfold_cross_validation")+ "/result/experiment_1/summary_of_experiment.txt";
    results.open (evaluation_file.c_str(), std::ofstream::trunc);

    results <<"system configuration:" 
            << "\n\t-experiment_name = " << name_of_approach
            << "\n\t-name_of_dataset = " << dataset//TODO make a parameter for this 
            //<< "\n\t-down_sampling_resolution = "<< 0.005* (exp_num -1)
            << "\n\t-number_of_bins = "<< number_of_bins
            << "\n\t-number_of_category = "<< "10"   
            << "\n\t-name_of_network = " << deep_learning_architecture
            << "\n\t-downsampling = " << downsampling_flag
            << "\n\t-downsampling_voxel_size = " << downsampling_voxel_size
            << "\n\t-image_normalization = " << image_normalization_flag
            << "\n\t-multiviews = " << multiviews_flag
            << "\n\t-pooling = " << pooling_flag
            << "\n\t-modelnet_dataset = " << modelnet_dataset ;

    results.close();
    

    ros::Time begin_Proc = ros::Time::now(); //start tic	

    
    /* ______________________
    |                       |
    |  Run train and test   |
    |_______________________| */ 

    int obj_no = 1;

    //// if modelnet dataset set k = 1, else k = 10
    size_t k_fold = 10; 
    if (modelnet_dataset)
    {    
        k_fold = 1; 
    }

    for (size_t iteration = 0; iteration < k_fold; iteration ++)
    {
        results.open (evaluation_file.c_str(), std::ofstream::app);
        results << "\n\nNo."<<"\tobject_name" <<"\t\t\t\t"<< "ground_truth" <<"\t"<< "prediction"<< "\t\t"<< "TP" << "\t"<< "FP"<< "\t"<< "FN \t\tdistance";
        results << "\n------------------------------------------------------------------------------------------------------------------------------------";
        results.close();


        //// generating test and train data for each iteration of k-fold cross-validation
        if (modelnet_dataset)
        {    
            modelNetTrainTestData(home_address);
        }
        else 
        {
            crossValidationDataCRC( k_fold, iteration, home_address);
            ROS_INFO("############################ FOLD %ld ############################", iteration + 1);
        }


        /* _______________________________
        |                                |
        |  conceptualize training data   |
        |________________________________| */

        ROS_INFO("\t\t[-]conceptualizing training data");  

        int track_id = 1;
        int view_id = 1;
       
        conceptualizingDeepLearningAndGoodDescriptor (  track_id, pp,
                                                        home_address, 
                                                        adaptive_support_lenght,
                                                        global_image_width,
                                                        threshold,
                                                        number_of_bins,
                                                        deep_learning_server,
                                                        downsampling, 
                                                        downsampling_voxel_size, 
                                                        modelnet_dataset /* NOTE: we do not need to perfom PCA for ModelNet*/   
                                                        );
        
        //// get list of all object categories
        vector <ObjectCategory> list_of_object_category = _pdb->getAllObjectCat();
        ROS_INFO(" %d categories exist in the perception database ", list_of_object_category.size() );

        //// initialize confusion_matrix
        if (!map_name_to_idx_flag)
        {  
            std::vector<int>  row (list_of_object_category.size(), 0);
            for (size_t i = 0; i < list_of_object_category.size(); i++ )
            {
                confusion_matrix.push_back (row);
                map_category_name_to_index.push_back(list_of_object_category.at(i).cat_name.c_str());
                map_name_to_idx_flag = true;
            }
        }

        /* ______________________________________________
        |                                                |
        |   read list of test data and check the system  |
        |________________________________________________| */
        
        //todo: path should be defined as a param
        test_csv.open ("/home/hamidreza/Desktop/test.csv", std::ofstream::trunc);

        //// read list of test data
        std::ifstream list_of_test_data (test_data_path.c_str());
        while (list_of_test_data.good ())// read categories address
        {
            string predict_category;
            std::string test_instance;
            std::getline (list_of_test_data, test_instance);
            if(test_instance.empty () || test_instance.at (0) == '#') // Skip blank lines or comments
                continue;
              
            
            //load a point cloud of object (*.pcd); PCD stands for point cloud data 
            std::string pcd_file_address; 
            pcd_file_address = home_address +"/"+ test_instance.c_str();
            boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);

            if (io::loadPCDFile <PointT> (pcd_file_address.c_str(), *target_pc) == -1)
            {	
                ROS_ERROR("\t\t[-]-Could not read given object %s :", pcd_file_address.c_str());
                return(0);
            }
            
            /* _______________________
            |                         |
            |  Object Representation  |
            |_________________________| */

            ros::Time begin_process = ros::Time::now(); //start tic	
            SITOV object_representation;

            /* ________________________
            |                          |
            |  GOOD shape description  |
            |__________________________| */

            boost::shared_ptr<pcl::PointCloud<PointT> > pca_object_view (new PointCloud<PointT>);
            boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
            vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
            double largest_side = 0;
            int  sign = 1;
            vector <float> view_point_entropy;
            string std_name_of_sorted_projected_plane;
            Eigen::Vector3f center_of_bbox;
            vector< float > object_description;          
           
            ///NOTE: for ModelNet, since all of pointclouds have been aligned, we do not need to perform PCA
            if (modelnet_dataset)
            {
                //// NOTE: since we do not need to compute the PCA of object, 
                //// we use the for_grasp function
                compuet_object_description_for_grasp( target_pc,
                                                      number_of_bins,
                                                      object_description );
            }	    
            else
            {
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
            }

            for (size_t i = 0; i < object_description.size(); i++)
            {
                object_representation.spin_image.push_back(object_description.at(i));
            
            }
            
            SITOV deep_representation_sitov;
            /// call deep learning service to represent the given GOOD description as vgg16  
            race_deep_learning_feature_extraction::deep_representation srv;
            srv.request.good_representation = object_representation.spin_image;
            if (deep_learning_server.call(srv))
            {
                pp.info(std::ostringstream().flush() << "################ receiving deep learning server response with the size of " << srv.response.deep_representation.size() );
                for (size_t i = 0; i < srv.response.deep_representation.size(); i++)
                {
                    deep_representation_sitov.spin_image.push_back(srv.response.deep_representation.at(i));
                    test_csv << srv.response.deep_representation.at(i) << "," ;
                }
            }
            else
            {
                ROS_ERROR("Failed to call deep learning service");
            }
            
            string tmp = pcd_file_address.substr(home_address.size(), pcd_file_address.size());
            string ground_truth = extractCategoryName(tmp);

            test_csv  << ground_truth << "\n";
        
            ROS_INFO("\t\t[-]size of object view histogram is = %d", object_representation.spin_image.size());          

            //get toc
            ros::Duration duration = ros::Time::now() - begin_process;
            double duration_sec = duration.toSec();
            ROS_INFO("\t\t[-]compute Object Representation for the given object took %f secs" , duration_sec);

            /* ______________________
            |                        |
            |   Object Recognition   |
            |________________________| */

            ros::Time start_time_recognition = ros::Time::now();
            
            vector <NOCD> nocd_object_category_distance;
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
                        vector<SITOV> objects_views_histograms = _pdb->getSITOVs(list_of_object_category.at(i).rtov_keys.at(j).c_str());
                        category_instances.push_back(objects_views_histograms.at(0));                      
                    }
                            
                    /* ________________________________________________________________________________________
                    |                                                                                          |
                    |   Compute the dissimilarity between the target object and all instances of a category    |
                    |__________________________________________________________________________________________| */
                    
                    /// xobjectCategoryDistance() -> returns minimum distance and best_matched_index 
                    float min_distance_object_category;
                    int best_matched_index;
                    float normalized_distance;

                    //// TODO: type of distance function can be a parameter 

                    //// euclidean distance
                    //euclideanBasedObjectCategoryDistance( deep_representation_sitov, category_instances, min_distance_object_category, best_matched_index, pp);
                    
                    //// chi-squared distance
                    //chiSquaredBasedObjectCategoryDistance( deep_representation_sitov, category_instances, min_distance_object_category, best_matched_index, pp);
                    
                    //// symmetric Kullback–Leibler divergence 
                    kLBasedObjectCategoryDistance( deep_representation_sitov, category_instances, min_distance_object_category, best_matched_index, pp);
                                
                    object_category_distance.push_back(min_distance_object_category);
                    //// the following msg can be used in computing top5 accuracy or other high-level process
                    NOCD tmp;
                    tmp.normalized_distance = min_distance_object_category;
                    tmp.cat_name = list_of_object_category.at(i).cat_name.c_str();
                    nocd_object_category_distance.push_back(tmp);

                }
                else 
                {
                    object_category_distance.push_back(1000000000);
                    //// the following msg can be used in computing top5 accuracy or other high-level process
                    NOCD tmp;
                    tmp.normalized_distance = 1000000000;
                    tmp.cat_name = list_of_object_category.at(i).cat_name.c_str();
                    nocd_object_category_distance.push_back(tmp);
                }
            }


            //// Timer toc
            duration = ros::Time::now() - beginProc;
            duration_sec = duration.toSec();

            /// Object Recognition 
            beginProc = ros::Time::now();
            float sigma_distance = 0;
            float recognition_threshold = 1000000000;
            int category_index = -1; 
            float minimum_distance = 9000000000;

            //simple nearest neighbor classifier
            findClosestCategory( object_category_distance,
                                 category_index, minimum_distance, pp, sigma_distance);
                
            std::string result_string;
            if (category_index == -1)
            {
                ROS_INFO("Predicted category is unknown");
                result_string = "unknown";
            }
            else
            {
                //ROS_INFO("Predicted category is %s", list_of_object_category.at(category_index).cat_name.c_str());
                result_string = list_of_object_category.at(category_index).cat_name.c_str();
            }
            
            ROS_INFO("\t\t[-]Object Recognition process took %f secs", (ros::Time::now() - start_time_recognition).toSec());

            //// RRTOV stands for Recognition results of Track Object View - it is a msg defined in race_perception_msgs
            RRTOV rrtov;
            rrtov.header.stamp = ros::Time::now();
            rrtov.track_id = track_id;
            rrtov.view_id = view_id;
            rrtov.recognition_result = result_string;
            rrtov.minimum_distance = minimum_distance;
            rrtov.ground_truth_name = pcd_file_address.c_str();
            rrtov.result = nocd_object_category_distance;

            evaluationFunction(rrtov);
        
            ros::spinOnce();
            track_id ++;
        }

        //pp.printCallback();

        list_of_test_data.close();
        reportCurrentResults(TPtmp, FPtmp, FNtmp, evaluation_file,0);
        ros::spinOnce();

        /* ________________________________
        |                                  |
        |   deconceptualizing train data   |
        |__________________________________| */
        
        deconceptualizingAllTrainData ();
    }
    results.close();
    
    //// get toc
    ros::Duration duration = ros::Time::now() - beginProc;
    reportAllExperiments (  TP,  FP,  FN, name_of_approach);
    
    double duration_sec = duration.toSec();
    
    reportCurrentResults(TP, FP, FN, evaluation_file,1);
    results.open (evaluation_file.c_str(), std::ofstream::app);
    results << "\n\t - This expriment took " << duration_sec << " secs\n\n";
    results << "========================================================================== \n"; 


    results << "\n\n - Confusion_matrix [" << confusion_matrix.size() << "," << confusion_matrix.at(0).size() << "]= \n\n";

    for (int i = 0; i < confusion_matrix.size(); i++ ) 
    {
        int sum_all_predicitions = 0;
        results << map_category_name_to_index.at(i) << ",\t\t";
        for (int j = 0; j < confusion_matrix.at(i).size(); j++ ) 
        {
            sum_all_predicitions += confusion_matrix.at(i).at(j);
            results << confusion_matrix.at(i).at(j) << ",\t";  
        }
        ROS_INFO ("confusion_matrix.at(i).at(i) = %d", confusion_matrix.at(i).at(i));
        ROS_INFO ("sum_all_predicitions = %d", sum_all_predicitions);

        float class_accuracy_tmp = float (confusion_matrix.at(i).at(i)) / float(sum_all_predicitions);
        results << "\t" << map_category_name_to_index.at(i) << " accuracy = " << class_accuracy_tmp;    
        results << "\n";
    }

    results << "\t\t";
    for (int i =0; i < confusion_matrix.size(); i++ ) 
    {
        results << map_category_name_to_index.at(i) << ", ";
    }

    results.close();

    results.close();
    ros::spinOnce();
    

    return 1;
}