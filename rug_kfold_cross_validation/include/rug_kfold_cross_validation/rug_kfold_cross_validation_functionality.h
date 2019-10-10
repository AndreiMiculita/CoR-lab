#ifndef _LEAVEONEOUT_EVALUATION_LIB_H_
#define _LEAVEONEOUT_EVALUATION_LIB_H_


/* _________________________________
  |                                 |
  |           Defines               |
  |_________________________________| */
#ifndef _LEAVEONEOUT_EVALUATION_LIB_DEBUG_
#define _LEAVEONEOUT_EVALUATION_LIB_DEBUG_ TRUE
#endif

/* _________________________________
   |                                 |
   |           INCLUDES              |
   |_________________________________| */

//Boost includes
#include <boost/make_shared.hpp>
//ROS includes
#include <ros/ros.h>

//PCL includes
#include <pcl/features/spin_image.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/features/esf.h>
#include <pcl/features/gfpfh.h>
#include <pcl/features/vfh.h>

//GRSD needs PCL 1.8.0
// #include <pcl/features/grsd.h>

//Eigen includes
#include <Eigen/Core>

//system includes
#include <std_msgs/String.h>
#include <sstream>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>    
#include <string.h>
#include <cctype> // string manupolation
#include <math.h> 
#include <cstring>


//package includes
#include <race_perception_msgs/perception_msgs.h>
#include <object_descriptor/object_descriptor_functionality.h>

//Gi includes
#include <pluginlib/class_list_macros.h> 
#include <stdio.h>
#include <race_perception_db/perception_db_serializer.h>
#include <race_perception_utils/cycle.h>
#include <race_perception_utils/print.h>


/* _________________________________
   |                                 |
   |           NAMESPACES            |
   |_________________________________| */

using namespace pcl;
using namespace race_perception_msgs;
using namespace race_perception_db;
using namespace race_perception_utils;

typedef PointXYZRGBA PointT; // define new type.


/* _________________________________
|                                 |
|        FUNCTION PROTOTYPES      |
|_________________________________| */



string extractObjectName (string object_name_orginal );

/////////////////////////////////////////////////////////////////////////////////////////////////////

string extractCategoryName (string instance_path);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int deleteObjectViewHistogramFromSpecificCategory(  std::string cat_name, unsigned int cat_id, 
													int track_id,  int view_id,
													PrettyPrint &pp);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int addObjectViewHistogramInSpecificCategory(std::string cat_name, unsigned int cat_id, 
        unsigned int track_id, unsigned int view_id, 
        SITOV objectViewHistogram , PrettyPrint &pp
        );

/////////////////////////////////////////////////////////////////////////////////////////////////////

int addObjectViewHistogramInSpecificCategoryDeepLearning(std::string cat_name, unsigned int cat_id, 
														unsigned int track_id, unsigned int view_id, 
														SITOV objectViewHistogram , PrettyPrint &pp
														);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizeObjectViewSpinImagesInSpecificCategory(std::string cat_name, unsigned int cat_id, 
        unsigned int track_id, unsigned int view_id, 
        vector <SITOV> SpinImageMsg , PrettyPrint &pp
        );

/////////////////////////////////////////////////////////////////////////////////////////////////////

int deleteObjectViewHistogramFromSpecificCategory(	std::string cat_name, unsigned int cat_id, 
													int track_id,  int view_id,
													SITOV objectViewHistogram , PrettyPrint &pp);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int crossValidationDataCRC(int K_fold, int iteration , string home_address);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int modelNetTrainTestData(string home_address);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int varingNumberOfCategories(int number_of_categories, string home_address);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int objectRepresentationBagOfWords (vector <SITOV> cluster_center, 
				    vector <SITOV> object_spin_images, 
				    SITOV  &object_representation );

/////////////////////////////////////////////////////////////////////////////////////////////////////

int notNormalizedObjectRepresentationBagOfWords( vector <SITOV> cluster_center, 
												 vector <SITOV> object_spin_images, 
												 SITOV  &object_representation );

/////////////////////////////////////////////////////////////////////////////////////////////////////

void delelteAllSITOVFromDB ();

/////////////////////////////////////////////////////////////////////////////////////////////////////

void delelteAllRVFromDB ();

/////////////////////////////////////////////////////////////////////////////////////////////////////

int deconceptualizingAllTrainData();

/////////////////////////////////////////////////////////////////////////////////////////////////////
 
int deleteObjectViewFromSpecificCategory(std::string cat_name, unsigned int cat_id, 
										 int track_id,  int view_id,
										 vector <SITOV> SpinImageMsg , PrettyPrint &pp);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int keypointSelection( boost::shared_ptr<pcl::PointCloud<PointT> > target_pc, 
						float uniform_sampling_size,
						boost::shared_ptr<pcl::PointCloud<PointT> > uniform_keypoints,
						boost::shared_ptr<pcl::PointCloud<int> > uniform_sampling_indices );

/////////////////////////////////////////////////////////////////////////////////////////////////////

// consider all features of the object 
int conceptualizingTrainData(int &track_id, 
							 PrettyPrint &pp,
							 string home_address,
							 int spin_image_width_int,
							 float spin_image_support_lenght_float,
							 size_t subsampled_spin_image_num_keypoints
							 );

/////////////////////////////////////////////////////////////////////////////////////////////////////

// consider only features of kepoints 
int conceptualizingTrainData2( int &track_id, 
							   PrettyPrint &pp,
							   string home_address,
							   int spin_image_width_int,
							   float spin_image_support_lenght_float,
							   float uniform_sampling_size
							  );

vector <SITOV>  readClusterCenterFromFile (string path);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingDictionaryBasedTrainData(int &track_id, 
			     PrettyPrint &pp,
			     string home_address,
			     int spin_image_width_int,
			     float spin_image_support_lenght_float,
			     double uniform_sampling_size, 
			     vector <SITOV> dictionary_of_spin_images
			    );

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingTrainDataBasedOnGenericAndSpecificDictionaries(int &track_id, 
			     PrettyPrint &pp,
			     string home_address,
			     int spin_image_width_int,
			     float spin_image_support_lenght_float,
			     double uniform_sampling_size, 
			     vector <SITOV> generic_dictionary
			    );

/////////////////////////////////////////////////////////////////////////////////////////////////////

int deconceptualizingDictionaryBasedTrainData( PrettyPrint &pp,
									string home_address,
									int spin_image_width_int,
									float spin_image_support_lenght_float,
									size_t subsampled_spin_image_num_keypoints );

/////////////////////////////////////////////////////////////////////////////////////////////////////

vector <int> generateSequence (int n);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int generateRrandomSequencesInstances ( string path,
				      					string home_address);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int generateRrandomSequencesCategories ( string home_address, 
										 int number_of_object_per_category);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int generateRrandomSequencesCategoriesKfold ( string home_address, 
											 int number_of_object_per_category);

/////////////////////////////////////////////////////////////////////////////////////////////////////

void reportCurrentResults(int TP, int FP, int FN,string fname, bool global);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int addingGaussianNoise (boost::shared_ptr<PointCloud<PointT> > input_pc, 
			 double standard_deviation,
			 boost::shared_ptr<PointCloud<PointT> > output_pc);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int addingGaussianNoiseXYZL (boost::shared_ptr<PointCloud<pcl::PointXYZL> > input_pc, 
			 double standard_deviation,
			 boost::shared_ptr<PointCloud<pcl::PointXYZL> > output_pc);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int downSampling ( boost::shared_ptr<PointCloud<PointT> > cloud, 		
		  double downsampling_voxel_size, 
		  boost::shared_ptr<PointCloud<PointT> > downsampled_pc);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int downSamplingXYZL ( boost::shared_ptr<PointCloud<pcl::PointXYZL>  > cloud, 		
		  float downsampling_voxel_size, 
		  boost::shared_ptr<PointCloud<pcl::PointXYZL> > downsampled_pc);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int estimateViewpointFeatureHistogram(boost::shared_ptr<PointCloud<PointT> > cloud, 
				    float normal_estimation_radius,
				    pcl::PointCloud<pcl::VFHSignature308>::Ptr &vfhs);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingVFHTrainData( int &track_id, 
								 PrettyPrint &pp,
								 string home_address, 
								 float normal_estimation_radius);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingVFHDownSampledTrainData( int &track_id, 
				PrettyPrint &pp,
				string home_address, 
				float normal_estimation_radius,
				float downsampling_voxel_size);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int estimateESFDescription (boost::shared_ptr<PointCloud<PointT> > cloud, 
			     pcl::PointCloud<pcl::ESFSignature640>::Ptr &esfs);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingESFTrainData( int &track_id, 
								 PrettyPrint &pp,
								 string home_address);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingESFDownSampledTrainData( int &track_id, 
					      PrettyPrint &pp,
					      string home_address,
					      float downsampling_voxel_size);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int estimateGRSDDescription( boost::shared_ptr<PointCloud<PointT> > cloud, 
							 float normal_estimation_radius,
							 pcl::PointCloud<pcl::GRSDSignature21>::Ptr &grsds);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingGRSDTrainData( int &track_id, 
				   PrettyPrint &pp,		  
				   string home_address,
				   float normal_estimation_radius);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingGRSDDownSampledTrainData( int &track_id, 
					      PrettyPrint &pp,
					      string home_address,
					      float normal_estimation_radius,
					      float downsampling_voxel_size);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int estimateGFPFH(boost::shared_ptr<PointCloud<pcl::PointXYZL> > cloud, 
		   pcl::PointCloud<pcl::GFPFHSignature16>::Ptr &gfpfhs);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingGFPFHTrainData( int &track_id, 
									PrettyPrint &pp,
									string home_address);


/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingGFPFHDownSampledTrainData( int &track_id, 
				    PrettyPrint &pp,
				    string home_address, 
				    float downsampling_voxel_siz );

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingDeepLearningAndGoodDescriptor( int &track_id, 
													PrettyPrint &pp,
													string home_address, 
													int adaptive_support_lenght,
													double global_image_width,
													int threshold,
													int number_of_bins,
													ros::ServiceClient deep_learning_server, 
													bool downsampling, 
													double downsampling_voxel_size,
													bool modelnet_dataset);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingDeepLearningAndGoodDescriptorPlusDataAugmentation( int &track_id, 
																		PrettyPrint &pp,
																		string home_address, 
																		int adaptive_support_lenght,
																		double global_image_width,
																		int threshold,
																		int number_of_bins,
																		ros::ServiceClient deep_learning_server, 
																		bool modelnet_dataset);
/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingGOODTrainData( int &track_id, 
								PrettyPrint &pp,
								string home_address, 
								int adaptive_support_lenght,
								double global_image_width,
								int threshold,
								int number_of_bins);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingGOODDownSampledTrainData( int &track_id, 
											PrettyPrint &pp,
											string home_address, 
											int adaptive_support_lenght,
											double global_image_width,
											int threshold,
											int number_of_bins,
											float downsampling_voxel_size );

/////////////////////////////////////////////////////////////////////////////////////////////////////
int conceptualizingTrainDataCRC( int &track_id, 
								 PrettyPrint &pp,
						      	 string home_address, 
								 int adaptive_support_lenght,
								 double global_image_width,
								 int threshold,
								 int number_of_bins);
								
/////////////////////////////////////////////////////////////////////////////////////////////////////

void writeToFile (string file_name, float value );

/////////////////////////////////////////////////////////////////////////////////////////////////////

bool fexists(std::string filename);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int ros_param_set (string topic_name, float value);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int set_parameters(string name_of_approach);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int numberOfPerformedExperiments(string name_of_approach, int &exp_num);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int reportAllExperiments (int TP, int FP, int FN,
						int number_of_bins, 
						string name_of_approach);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int reportAllExperiments (int TP, int FP, int FN,
			     string name_of_approach);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int reportAllDeepLearningExperiments ( int TP, int FP, int FN,
											string dataset,
											int number_of_bins, 
											string name_of_network,
											string multi_view_flag,
											string image_normalization_flag,
											string pooling_flag,
											bool downsampling,
											double downsampling_voxel_size,
											string name_of_approach,
											double global_class_accuracy);

////////////////////////////////////////////////////////////////////////////////////////////////////

int chiSquaredDistanceBetweenTwoObjectViewHistogram( SITOV objectViewHistogram1,
													 SITOV objectViewHistogram2, 
													 float &diffrence);

////////////////////////////////////////////////////////////////////////////////////////////////////

int chiSquaredBasedObjectCategoryDistance( 	SITOV target,
											vector< SITOV > category_instances,
											float &minimumDistance, 
											int &best_matched_index, 
											PrettyPrint &pp);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int squaredChordDistanceBetweenTwoObjectViewHistogram( SITOV objectViewHistogram1,
													 SITOV objectViewHistogram2, 
													 float &diffrence);

////////////////////////////////////////////////////////////////////////////////////////////////////

int squaredChordBasedObjectCategoryDistance( 	SITOV target,
											vector< SITOV > category_instances,
											float &minimumDistance, 
											int &best_matched_index, 
											PrettyPrint &pp);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int euclideanBasedObjectCategoryDistance(	SITOV target,
											vector< SITOV > category_instances,
											float &minimumDistance, 
											int &best_matched_index, 
											PrettyPrint &pp);

/////////////////////////////////////////////////////////////////////////////////////////////////////

int kLBasedObjectCategoryDistance(  SITOV target,
									vector< SITOV > category_instances,
									float &minimumDistance, 
									int &best_matched_index, 
									PrettyPrint &pp);
/////////////////////////////////////////////////////////////////////////////////////////////////////added A&K
int FidelityBasedObjectCategoryDistance(  SITOV target,
									vector< SITOV > category_instances,
									float &minimumDistance, 
									int &best_matched_index, 
									PrettyPrint &pp);
/////////////////////////////////////////////////////////////////////////////////////////////////////added A&K
int FidelityDistanceBetweenTwoObjectViewHistogram( SITOV objectViewHistogram1,
													 SITOV objectViewHistogram2, 
													 float &diffrence);
/////////////////////////////////////////////////////////////////////////////////////////////////////

void  confusionMatrixGenerator( string true_category, string predicted_category, 
								std::vector<string> map_category_name_to_index,
								std::vector< std::vector <int> > &confusion_matrix );

/////////////////////////////////////////////////////////////////////////////////////////////////////

void findClosestCategory( vector<float> object_category_distances,
						  int &cat_index, 
						  float &mindist, 
						  PrettyPrint &pp, 
						  float &sigma_distance);



#endif

