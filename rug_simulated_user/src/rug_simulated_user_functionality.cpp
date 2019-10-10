// ############################################################################
//    
//   Created: 	1/09/2014
//   Author : 	Hamidreza Kasaei
//   Email  :	seyed.hamidreza@ua.pt
//   Purpose: 	This program follows the teaching protocol and autonomously
//		interact with the system using teach, ask and correct actions. 
// 		For each newly taught category, the average sucess of the system
// 		should be computed. To do that, the simulated teacher repeatedly 
// 		picks object views of the currently known categories from a 
// 		database and presents them to the system for checking whether 
// 		the system can recognize them. If not, the simulated teacher provides
// 		corrective feedback.
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
   |           INCLUDES              |
   |_________________________________| */

   //ROS includes
#include <ros/ros.h>
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/common/transforms.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <visualization_msgs/Marker.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/visualization/cloud_viewer.h>
#include <ros/package.h>
#include <pcl/io/ply_io.h>
#include <tf/tf.h>


//system includes
#include <std_msgs/String.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>
#include <CGAL/Plane_3.h>
#include <algorithm>


//ros includes 
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>

//perception db includes
#include <race_perception_msgs/perception_msgs.h>
#include <race_perception_db/perception_db.h>
#include <race_perception_db/perception_db_serializer.h>

//pcl includes

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ros/conversions.h>
#include <pcl/io/pcd_io.h>

// Need to include the pcl ros utilities

//package includes
#include <object_descriptor/object_descriptor_functionality.h>
#include <feature_extraction/spin_image.h>
#include <race_3d_object_tracking/TrackedObjectPointCloud.h>
#include <object_conceptualizer/object_conceptualization.h>
#include <race_perception_utils/print.h>

// #include <race_deep_learning_feature_extraction/vgg16_model.h>
#include <race_deep_learning_feature_extraction/deep_representation.h>

/* _________________________________
  |                                 |
  |            constant            |
  |_______________________________| */
// #define spin_image_width 4
// #define subsample_spinimages 0
// #define spin_image_support_lenght 0.05

#define recognitionThershold 20000


/* _________________________________
  |                                 |
  |         Global variable         |
  |_________________________________| */

  
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointXYZRGBA T;  

PerceptionDB* _pdb; //initialize the class
int track_id_gloabal =1;
ofstream allFeatures;
std::string PCDFileAddressTmp;
int Hist_lenght = 0;

// std::string home_directory_address = "/home/hamidreza/";
std::string home_directory_address= "/home/cor/datasets/washington_RGBD_object/";


using namespace pcl;
using namespace std;
using namespace ros;


string fixedLength (string name , size_t length)
{
	while (name.length() < length)
    { 
		name += " ";
    }
    return (name);
} 


string extractObjectNameSimulatedUser (string object_name_orginal )
{
    std:: string object_name;
    std:: string tmp_object_name = object_name_orginal;// for writing object name in result file;
    int rfind = tmp_object_name.rfind("//")+2;       
    int len = tmp_object_name.length();

    object_name = object_name_orginal.substr(rfind, tmp_object_name.length());

	while (object_name.length() < 30)
    { 
		object_name += " ";
    }

    return (object_name);
} 
 
string extractCategoryName (string InstancePath )
{
//  ROS_INFO("\t\t instance_path = %s", InstancePath.c_str());
    string categoryName="";	    
    int ffind = InstancePath.find("//")+2;  
//    ROS_INFO("\t\t left = %d", ffind);
    int lfind =  InstancePath.find("_Cat");       
//  ROS_INFO("\t\t right = %d", lfind);

    for (int i=0; i<(lfind-ffind); i++)
    {
		categoryName += InstancePath.at(i+ffind);
    }
    return (categoryName);
}


int updateNaiveBayesModel(std::string cat_name,
			  unsigned int cat_id,
			  PrettyPrint &pp)
{    
    
  /* _____________________________________
    |                                     |
    |   get list of all object categories |
    |_____________________________________| */

    vector <ObjectCategory> ListOfObjectCategory = _pdb->getAllObjectCat();
    pp.info(std::ostringstream().flush() << ListOfObjectCategory.size()<<" categories exist in the perception database" );
    
    
    /* _____________________________________________
    |                        	                  |
    |   compute total number of training object  |
    |___________________________________________| */
    
    int totoal_number_of_training_data =0;
    for (int i = 0; i < ListOfObjectCategory.size(); i++) // all category exist in the database 
    {
		totoal_number_of_training_data += ListOfObjectCategory.at(i).rtov_keys.size();
    }
    pp.info(std::ostringstream().flush() << "Total number of training data = " << totoal_number_of_training_data);
    
    /* _____________________________________________________
      |                                                    |
      |    create a naive Bayes model for each category   |
      |__________________________________________________| */
	    
    string package_path  = ros::package::getPath("race_naive_bayes_object_recognition");
    for (size_t i = 0; i < ListOfObjectCategory.size(); i++) // all category exist in the database 
    {
 	
	if (ListOfObjectCategory.at(i).rtov_keys.size() > 1) 
	{    
	    vector <float> overal_category_representation;
	    float total_number_of_words_in_category=0;
	    vector <float> likelihoods_of_all_words;
	    vector< SITOV > tmp_obj= _pdb->getSITOVs(ListOfObjectCategory.at(0).rtov_keys.at(0).c_str());
	    int size_of_dictionary = tmp_obj.at(0).spin_image.size();
	    for (int k =0; k < size_of_dictionary; k++)
	    {
			overal_category_representation.push_back(0);
			likelihoods_of_all_words.push_back(0);
	    }  
	    
	    pp.info(std::ostringstream().flush() << ListOfObjectCategory.at(i).cat_name.c_str() <<" category has " 
						 << ListOfObjectCategory.at(i).rtov_keys.size()<< " views");
	    /* _____________________________________________________________________________
	    |                                     					   |
	    |   overal_category_representation : total_number_of_each_words_in_category   |
	    |____________________________________________________________________________| */

	    std::vector< SITOV > category_instances; 

	    if (strcmp (cat_name.c_str(),ListOfObjectCategory.at(i).cat_name.c_str())==0)
	    {
		for (size_t j = 0; j < ListOfObjectCategory.at(i).rtov_keys.size(); j++)
		{
		    vector< SITOV > objectViewHistogram = _pdb->getSITOVs(ListOfObjectCategory.at(i).rtov_keys.at(j).c_str());
		    category_instances.push_back(objectViewHistogram.at(0));
				    
		    for (int k =0; k < objectViewHistogram.at(0).spin_image.size(); k++)
		    {
			overal_category_representation.at(k) += objectViewHistogram.at(0).spin_image.at(k);
			total_number_of_words_in_category += objectViewHistogram.at(0).spin_image.at(k);
		    }
		    
		}
    // 		pp.info(std::ostringstream().flush() << "size of object view histogram = " << objectViewHistogram.at(0).spin_image.size());
    // 		pp.info(std::ostringstream().flush() << "size of overal_category_representation = " << overal_category_representation.size());
		/* _____________________________________
		|                                      |
		|   compute likelihoods_of_all_words  |
		|____________________________________| */
				
		for (int idx = 0; idx <  size_of_dictionary; idx++)
		{
		    likelihoods_of_all_words.at(idx) = float(1+overal_category_representation.at(idx)) /
						    float(size_of_dictionary+total_number_of_words_in_category);							    

		}

		/* _____________________________________________
		|                                              |
		|   write the likelihoods model to the files  |
		|____________________________________________| */
		
		cout << "category name = "<<ListOfObjectCategory.at(i).cat_name ;
		
		string systemStringCommand= "mkdir "+ package_path+ "/NaiveBayesModels/"+ ListOfObjectCategory.at(i).cat_name ;
		system( systemStringCommand.c_str());
		
		
		string likelihoods_path = package_path + "/NaiveBayesModels/" + ListOfObjectCategory.at(i).cat_name + "/likelihoods.txt";
		ROS_INFO("\t\t[-]- likelihoods_path = %s", likelihoods_path.c_str());
		
		std::ofstream likelihoods (likelihoods_path.c_str(), std::ofstream::trunc);
  
		for (int i=0; i < likelihoods_of_all_words.size(); i++)
		{
		    likelihoods << likelihoods_of_all_words.at(i)<<"\n";
		}
		likelihoods.close();
	    }
	    
	    /* _____________________________________________
	    |                                              |
	    |  	   write the priors model to the files    |
	    |____________________________________________| */
	    
	    string prior_probability_path = package_path + "/NaiveBayesModels/" + ListOfObjectCategory.at(i).cat_name + "/prior_probability.txt";		
	    ROS_INFO("\t\t[-]- prior_probability_path = %s", prior_probability_path.c_str());
	    std::ofstream prior_probability (prior_probability_path.c_str(), std::ofstream::trunc);  
	    float prior_probability_value = float (ListOfObjectCategory.at(i).rtov_keys.size())/float (totoal_number_of_training_data);
	    prior_probability << prior_probability_value ;
	    prior_probability.close();	
	    ROS_INFO("Prior probability = %f", prior_probability_value);	    
	}
    }
  
    return (1);
}


int updateGOODNaiveBayesModel(std::string cat_name,
			       unsigned int cat_id,
			       PrettyPrint &pp)
{    
    
  /* _____________________________________
    |                                     |
    |   get list of all object categories |
    |_____________________________________| */

    vector <ObjectCategory> ListOfObjectCategory = _pdb->getAllObjectCat();
    pp.info(std::ostringstream().flush() << ListOfObjectCategory.size()<<" categories exist in the perception database" );
    
    
    /* _____________________________________________
    |                        	                  |
    |   compute total number of training object  |
    |___________________________________________| */
    
    int totoal_number_of_training_data =0;
    for (int i = 0; i < ListOfObjectCategory.size(); i++) // all category exist in the database 
    {
	totoal_number_of_training_data += ListOfObjectCategory.at(i).rtov_keys.size();
    }
    pp.info(std::ostringstream().flush() << "Total number of training data =" << totoal_number_of_training_data);
    
    /* _____________________________________________________
      |                                                    |
      |    create a naive Bayes model for each category   |
      |__________________________________________________| */
	    
    string package_path  = ros::package::getPath("race_naive_bayes_object_recognition");
    for (size_t i = 0; i < ListOfObjectCategory.size(); i++) // all category exist in the database 
    {
 	
	if (ListOfObjectCategory.at(i).rtov_keys.size() > 1) 
	{    
	    vector <float> overal_category_representation;
	    float total_number_of_bins_in_category=0;
	    vector <float> likelihoods_of_all_words;
	    vector< SITOV > tmp_obj= _pdb->getSITOVs(ListOfObjectCategory.at(0).rtov_keys.at(0).c_str());
	    int size_of_dictionary = tmp_obj.at(0).spin_image.size();
	    for (int k =0; k < size_of_dictionary; k++)
	    {
		overal_category_representation.push_back(0);
		likelihoods_of_all_words.push_back(0);
	    }  
	    
	    pp.info(std::ostringstream().flush() << ListOfObjectCategory.at(i).cat_name.c_str() <<" category has " 
						 << ListOfObjectCategory.at(i).rtov_keys.size()<< " views");
	    /* _____________________________________________________________________________
	    |                                     					   |
	    |   overal_category_representation : total_number_of_each_words_in_category   |
	    |____________________________________________________________________________| */

	    std::vector< SITOV > category_instances; 

	    if (strcmp (cat_name.c_str(),ListOfObjectCategory.at(i).cat_name.c_str())==0)
	    {
		for (size_t j = 0; j < ListOfObjectCategory.at(i).rtov_keys.size(); j++)
		{
		    vector< SITOV > objectViewHistogram = _pdb->getSITOVs(ListOfObjectCategory.at(i).rtov_keys.at(j).c_str());
		    category_instances.push_back(objectViewHistogram.at(0));
				    
		    for (int k =0; k < objectViewHistogram.at(0).spin_image.size(); k++)
		    {
			overal_category_representation.at(k) += objectViewHistogram.at(0).spin_image.at(k);
			//total_number_of_bins_in_category ++;
		    }
		    
		}
    // 		pp.info(std::ostringstream().flush() << "size of object view histogram = " << objectViewHistogram.at(0).spin_image.size());
    // 		pp.info(std::ostringstream().flush() << "size of overal_category_representation = " << overal_category_representation.size());
		/* _____________________________________
		|                                      |
		|   compute likelihoods_of_all_words  |
		|____________________________________| */
				
		for (int idx = 0; idx <  size_of_dictionary; idx++)
		{
		    likelihoods_of_all_words.at(idx) = 0.0001+ float(overal_category_representation.at(idx)) /
						    float( ListOfObjectCategory.at(i).rtov_keys.size());	/// the sumation of all the probability of a GOOD descriptor is 3 						    

		}

		/* _____________________________________________
		|                                              |
		|   write the likelihoods model to the files  |
		|____________________________________________| */
		
		cout << "category name = "<<ListOfObjectCategory.at(i).cat_name ;
		
		string systemStringCommand= "mkdir "+ package_path+ "/NaiveBayesModels/"+ ListOfObjectCategory.at(i).cat_name ;
		system( systemStringCommand.c_str());
		
		
		string likelihoods_path = package_path + "/NaiveBayesModels/" + ListOfObjectCategory.at(i).cat_name + "/likelihoods.txt";
		ROS_INFO("\t\t[-]- likelihoods_path = %s", likelihoods_path.c_str());
		
		std::ofstream likelihoods (likelihoods_path.c_str(), std::ofstream::trunc);
  
		for (int i=0; i < likelihoods_of_all_words.size(); i++)
		{
		    likelihoods << likelihoods_of_all_words.at(i)<<"\n";
		}
		likelihoods.close();
	    }
	    
	    /* _____________________________________________
	    |                                              |
	    |  	   write the priors model to the files    |
	    |____________________________________________| */
	    
	    string prior_probability_path = package_path + "/NaiveBayesModels/" + ListOfObjectCategory.at(i).cat_name + "/prior_probability.txt";		
	    ROS_INFO("\t\t[-]- prior_probability_path = %s", prior_probability_path.c_str());
	    std::ofstream prior_probability (prior_probability_path.c_str(), std::ofstream::trunc);  
	    float prior_probability_value = float (ListOfObjectCategory.at(i).rtov_keys.size())/float (totoal_number_of_training_data);
	    prior_probability << prior_probability_value ;
	    prior_probability.close();	
	    ROS_INFO("Prior probability = %f", prior_probability_value);	    
	}
    }
  
    return (1);
}



int putObjectViewSpinImagesInSpecificCategory(std::string cat_name, unsigned int cat_id, 
					    unsigned int track_id, unsigned int view_id, 
					    boost::shared_ptr< vector <SITOV> > SpinImageMsg,
					    double &Cat_ICD )
{
	PrettyPrint pp;
	SITOV msg_in;
	RTOV _rtov;
	_rtov.track_id = track_id;
	_rtov.view_id = view_id;
	
	for (size_t i = 0; i < SpinImageMsg->size(); i++)
	{
	    msg_in = SpinImageMsg->at(i);
	    msg_in.spin_img_id = i;

	    uint32_t sp_size = ros::serialization::serializationLength(msg_in);
	    
	    boost::shared_array<uint8_t> sp_buffer(new uint8_t[sp_size]);
	    PerceptionDBSerializer<boost::shared_array<uint8_t>, SITOV>::serialize(sp_buffer, msg_in, sp_size);
	    leveldb::Slice sp_s((char*)sp_buffer.get(), sp_size);
	    std::string sp_key = _pdb->makeSIKey(key::SI, track_id, view_id, i );

	    //Put slice to the db
	    _pdb->put(sp_key, sp_s); 
	    
	    //create a list of key of spinimage
	    _rtov.sitov_keys.push_back(sp_key);
	    
	}
	uint32_t v_size = ros::serialization::serializationLength(_rtov);

	boost::shared_array<uint8_t> v_buffer(new uint8_t[v_size]);
	PerceptionDBSerializer<boost::shared_array<uint8_t>, RTOV>::serialize(v_buffer, _rtov, v_size);	
	
	leveldb::Slice v_s((char*)v_buffer.get(), v_size);
	
	std::string v_key = _pdb->makeKey(key::RV, track_id, view_id);
	ROS_INFO("\t\t[-]v_key: %s, track_id: %i, view_id: %i", v_key.c_str(), track_id, view_id);

	//Put one view to the db
	_pdb->put(v_key, v_s);
	
	////////////////////////////////////////////////////////////////
	//Put OC with one view
	ObjectCategory _oc;
	
	std::string oc_key = _pdb->makeOCKey(key::OC, cat_name, cat_id);

	std::string str_oc;
	_pdb->get(oc_key, &str_oc);
	uint32_t oc_size = str_oc.length();
	if (oc_size != 0) //Object category exist.
	{
	    boost::shared_array<uint8_t> oc_dbuffer(new uint8_t[oc_size]);
	    memcpy(oc_dbuffer.get(), str_oc.data(), str_oc.size()); //Maybe a bug in ROS:: without this memcpy, runtime error occurs!!!!!!

	    //deserialize Msg 
	    race_perception_db::PerceptionDBSerializer<boost::shared_array<uint8_t>, race_perception_msgs::ObjectCategory>::deserialize(oc_dbuffer, _oc, oc_size);
	}
	    
	_oc.cat_name = cat_name;
	_oc.cat_id = cat_id ;
	_oc.rtov_keys.push_back(v_key);
	
	// when a new object view add to database, ICD should be update
	vector <  vector <SITOV> > category_instances;
	for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
	{
		 vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
		 category_instances.push_back(objectViewSpinimages);
	}
	
	double New_ICD = 0;
	intraCategoryDistance(category_instances, New_ICD, pp);
// 	ROS_INFO("\t\t[-]- ICD = %f", New_ICD);
	_oc.icd = New_ICD;
	
	
	Cat_ICD = New_ICD;
	oc_size = ros::serialization::serializationLength(_oc);

	boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
	PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
	leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
	_pdb->put(oc_key, ocs);
	return (0);
}



int IntroduceNewInstance ( std::string PCDFileAddress,
			   unsigned int cat_id, 
			   unsigned int track_id,
			   unsigned int view_id, 
			   int spin_image_width_int,
			   float spin_image_support_lenght_float,
			   size_t subsampled_spin_image_num_keypoints
		    	 )
{

//     string categoryName = PCDFileAddress;
//     categoryName.resize(13);
    
    string categoryName = extractCategoryName(PCDFileAddress);

    
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#' || categoryName == "Category//Unk") // Skip blank lines or comments
    {
	return 0;
    }
    
    PCDFileAddress = home_directory_address +"/"+ PCDFileAddress.c_str();

    //load a PCD object  
    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
    if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *PCDFile) == -1)
    {	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
    }
    else
    {
	    ROS_INFO("\t\t[1]-Loaded a point cloud: %s", PCDFileAddress.c_str());
    }
    
    //compute Spin Image for given point clould
    //Declare a boost share ptr to the SITOV msg
    boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
    objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
    
    //Call the library function for estimateSpinImages
    estimateSpinImages(PCDFile, 
		    0.01 /*downsampling_voxel_size*/, 
		    0.05 /*normal_estimation_radius*/,
		    spin_image_width_int    /*spin_image_width*/,
		    0.0 /*spin_image_cos_angle*/,
		    1   /*spin_image_minimum_neighbor_density*/,
		    spin_image_support_lenght_float /*spin_image_support_lenght*/,
		    objectViewSpinImages,
		    subsampled_spin_image_num_keypoints /*subsample spinimages*/
		    );
			    

    double Cat_ICD=0.0001;
    putObjectViewSpinImagesInSpecificCategory(categoryName,cat_id,track_id,view_id,objectViewSpinImages,Cat_ICD);

    ROS_INFO("\t\t[-]-%s created...",categoryName.c_str());
        
    return (0);
}


int IntroduceNewInstance2 ( string database_path,
			    std::string PCDFileAddress,
			    unsigned int cat_id, 
			    unsigned int track_id,
			    unsigned int view_id, 
			    int spin_image_width_int,
			    float spin_image_support_lenght_float,
			    float uniform_sampling_size
			    )
{

//     string categoryName = PCDFileAddress;
//     categoryName.resize(13);
    ROS_INFO ("name of given object view = %s",PCDFileAddress.c_str());
    string categoryName = extractCategoryName(PCDFileAddress);

    
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#' || categoryName == "Category//Unk") // Skip blank lines or comments
    {
	return 0;
    }
    
    PCDFileAddress = database_path +"/"+ PCDFileAddress.c_str();

    //load a PCD object  
    boost::shared_ptr<PointCloud<PointT> > test_point_cloud (new PointCloud<PointT>);
    if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *test_point_cloud) == -1)
    {	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
    }
    else
    {
	    ROS_INFO("\t\t[1]-IntroduceNewInstance2: Loaded a point cloud: %s", PCDFileAddress.c_str());
    }
       
    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
    pcl::VoxelGrid<PointT > voxelized_point_cloud;	
    voxelized_point_cloud.setInputCloud (test_point_cloud);
    voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
    voxelized_point_cloud.filter (*PCDFile);
       
    /* ________________________________________________
    |                                                 |
    |  Compute the Spin-Images for given point cloud  |
    |_________________________________________________| */
    //Declare a boost share ptr to the spin image msg	
    boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
    objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
    
    boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
    boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
    keypoint_selection( PCDFile, 
			uniform_sampling_size,
			uniform_keypoints,
			uniform_sampling_indices);
    
    estimateSpinImages2(PCDFile, 
			    0.01 /*downsampling_voxel_size*/, 
			    0.05 /*normal_estimation_radius*/,
			    spin_image_width_int /*spin_image_width*/,
			    0.0 /*spin_image_cos_angle*/,
			    1 /*spin_image_minimum_neighbor_density*/,
			    spin_image_support_lenght_float /*spin_image_support_lenght*/,
			    objectViewSpinImages,
			    uniform_sampling_indices /*subsample spinimages*/
	);
// 	{
// 	    pp.error(std::ostringstream().flush() << "Could not compute spin images");
// 	    return (0);
// 	}
    
    ROS_INFO("Given object view has %d spin images", objectViewSpinImages->size() );


    double Cat_ICD=0.0001;
    putObjectViewSpinImagesInSpecificCategory(categoryName,cat_id,track_id,view_id,objectViewSpinImages,Cat_ICD);

    ROS_INFO("\t\t[-]-%s *******************************************created...",categoryName.c_str());
        
    return (0);
}



int addObjectViewHistogramInSpecificCategory(std::string cat_name,
					      unsigned int cat_id, 
					      unsigned int track_id,
					      unsigned int view_id, 
					      SITOV objectViewHistogram,
					      PrettyPrint &pp )
{
    //PrettyPrint pp;
    SITOV msg_in;
    RTOV _rtov;
    _rtov.track_id = track_id;
    _rtov.view_id = view_id;

//     for (size_t i = 0; i < SpinImageMsg.size(); i++)
//     {
        msg_in = objectViewHistogram;
        msg_in.spin_img_id = 1;

        uint32_t sp_size = ros::serialization::serializationLength(msg_in);

        boost::shared_array<uint8_t> sp_buffer(new uint8_t[sp_size]);
        PerceptionDBSerializer<boost::shared_array<uint8_t>, SITOV>::serialize(sp_buffer, msg_in, sp_size);
        leveldb::Slice sp_s((char*)sp_buffer.get(), sp_size);
        std::string sp_key = _pdb->makeSIKey(key::SI, track_id, view_id, 1 );

        //Put slice to the db
        _pdb->put(sp_key, sp_s); 

        //create a list of key of spinimage
        _rtov.sitov_keys.push_back(sp_key);

//     }

    uint32_t v_size = ros::serialization::serializationLength(_rtov);

    boost::shared_array<uint8_t> v_buffer(new uint8_t[v_size]);
    PerceptionDBSerializer<boost::shared_array<uint8_t>, RTOV>::serialize(v_buffer, _rtov, v_size);	

    leveldb::Slice v_s((char*)v_buffer.get(), v_size);

    std::string v_key = _pdb->makeKey(key::RV, track_id, view_id);
    ROS_INFO("\t\t [-] v_key: %s, view_id: %i, track_id: %i", v_key.c_str(), view_id, track_id);

    //Put one view to the db
    _pdb->put(v_key, v_s);

    ObjectCategory _oc;

    std::string oc_key = _pdb->makeOCKey(key::OC, cat_name, cat_id);

    std::string str_oc;
    _pdb->get(oc_key, &str_oc);
    uint32_t oc_size = str_oc.length();
    if (oc_size != 0) //Object category exist.
    {
        boost::shared_array<uint8_t> oc_dbuffer(new uint8_t[oc_size]);
        memcpy(oc_dbuffer.get(), str_oc.data(), str_oc.size()); //Maybe a bug in ROS:: without this memcpy, runtime error occurs!!!!!!

        //deserialize Msg 
        race_perception_db::PerceptionDBSerializer<boost::shared_array<uint8_t>, race_perception_msgs::ObjectCategory>::deserialize(oc_dbuffer, _oc, oc_size);
    }

    _oc.cat_name = cat_name;
    _oc.cat_id = cat_id ;
    _oc.rtov_keys.push_back(v_key);

    // when a new object view add to database, ICD should be update
    // vector <  vector <SITOV> > category_instances;
    // for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
    // {
    //     vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
    //     category_instances.push_back(objectViewSpinimages);
    // }

    // double New_ICD = 0;
    // intraCategoryDistance(category_instances, New_ICD, pp);
    // _oc.icd = New_ICD;

	_oc.icd = 0.00001;

    oc_size = ros::serialization::serializationLength(_oc);

    pp.info(std::ostringstream().flush() << _oc.cat_name.c_str() << " category has " << _oc.rtov_keys.size() << " objects.");
    //pp.info(std::ostringstream().flush() << "ICD for " << _oc.cat_name.c_str() << " category updated. New ICD is: "<< _oc.icd);

    boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
    PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
    leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
    _pdb->put(oc_key, ocs);

    return (1);
}

int addObjectViewHistogramInSpecificCategoryDeepLearning(std::string cat_name, unsigned int cat_id, 
															unsigned int track_id, unsigned int view_id, 
															SITOV objectViewHistogram , PrettyPrint &pp
															)
{
	//ROS_INFO("****========*********========*********========*********========*********========*****");

	//// ADD A NEW OBJECT Feature TO PDB
    SITOV msg_in;
    RTOV _rtov;
    _rtov.track_id = track_id;
    _rtov.view_id = view_id;

	msg_in = objectViewHistogram;
	msg_in.spin_img_id = 1;


	ros::Time start_time = ros::Time::now();

	uint32_t sp_size = ros::serialization::serializationLength(msg_in);

	boost::shared_array<uint8_t> sp_buffer(new uint8_t[sp_size]);
	PerceptionDBSerializer<boost::shared_array<uint8_t>, SITOV>::serialize(sp_buffer, msg_in, sp_size);
	leveldb::Slice sp_s((char*)sp_buffer.get(), sp_size);
	std::string sp_key = _pdb->makeSIKey(key::SI, track_id, view_id, 1 );

	//Put slice to the db
	_pdb->put(sp_key, sp_s); 

	//ROS_INFO("****========***** add a FEATURE to PDB took = %f", (ros::Time::now() - start_time).toSec());


	//// ADD A NEW OBJECT VIEW TO PDB
	start_time = ros::Time::now();

	//create a list of key of spinimage
	_rtov.sitov_keys.push_back(sp_key);

    uint32_t v_size = ros::serialization::serializationLength(_rtov);

    boost::shared_array<uint8_t> v_buffer(new uint8_t[v_size]);
    PerceptionDBSerializer<boost::shared_array<uint8_t>, RTOV>::serialize(v_buffer, _rtov, v_size);	

    leveldb::Slice v_s((char*)v_buffer.get(), v_size);

    std::string v_key = _pdb->makeKey(key::RV, track_id, view_id);
    ROS_INFO("\t\t[-]v_key: %s, view_id: %i, track_id: %i", v_key.c_str(), view_id, track_id);

    //Put one view to the db
    _pdb->put(v_key, v_s);
	//ROS_INFO("****========***** add a VIEW to PDB took = %f", (ros::Time::now() - start_time).toSec());


	//// UDPDATE CATEGORY TO PDB
	start_time = ros::Time::now();

    ObjectCategory _oc;

    std::string oc_key = _pdb->makeOCKey(key::OC, cat_name, cat_id);

    std::string str_oc;
    _pdb->get(oc_key, &str_oc);
    uint32_t oc_size = str_oc.length();
    if (oc_size != 0) //Object category exist.
    {
        boost::shared_array<uint8_t> oc_dbuffer(new uint8_t[oc_size]);
        memcpy(oc_dbuffer.get(), str_oc.data(), str_oc.size()); //Maybe a bug in ROS:: without this memcpy, runtime error occurs!!!!!!

        //deserialize Msg 
        race_perception_db::PerceptionDBSerializer<boost::shared_array<uint8_t>, race_perception_msgs::ObjectCategory>::deserialize(oc_dbuffer, _oc, oc_size);
    }

    _oc.cat_name = cat_name;
    _oc.cat_id = cat_id ;
    _oc.rtov_keys.push_back(v_key);

    // when a new object view add to database, ICD should be update
//     vector <  vector <SITOV> > category_instances;
//     for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
//     {
//         vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
//         category_instances.push_back(objectViewSpinimages);
//     }

    // std::vector< SITOV > category_instances; 
    // for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
    // {
	// 	vector< SITOV > objectViewHistogram = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
	// 	category_instances.push_back(objectViewHistogram.at(0));
	// 	// pp.info(std::ostringstream().flush() << "size of object view histogram = " << objectViewHistogram.size());
	// 	// pp.info(std::ostringstream().flush() << "key for the object view histogram = " << v_oc.at(i).rtov_keys.at(j).c_str());
    // }

    
    float New_ICD = 1;
//     intraCategoryDistance(category_instances, New_ICD, pp);
    //histogramBasedIntraCategoryDistance(category_instances,New_ICD,pp);
    _oc.icd = New_ICD;

    oc_size = ros::serialization::serializationLength(_oc);

    pp.info(std::ostringstream().flush() << _oc.cat_name.c_str() << " category has " << _oc.rtov_keys.size() << " objects.");
    //pp.info(std::ostringstream().flush() << "ICD for " << _oc.cat_name.c_str() << " category updated. New ICD is: "<< _oc.icd);

    boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
    PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
    leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
    _pdb->put(oc_key, ocs);
	//ROS_INFO("****========***** update a CATEGORY to PDB took = %f", (ros::Time::now() - start_time).toSec());
	//ROS_INFO("****========*********========*********========*********========*********========*****");
    return (1);
}

int introduceNewInstanceGOOD ( string database_path,
								std::string PCDFileAddress,
								unsigned int cat_id, 
								unsigned int track_id,
								unsigned int view_id, 
								int adaptive_support_lenght,
								double global_image_width,
								int threshold,
								int number_of_bins,
								PrettyPrint &pp
								)
{

    ROS_INFO ("name of given object view = %s",PCDFileAddress.c_str());
    string categoryName = extractCategoryName(PCDFileAddress);

    
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#') // Skip blank lines or comments
    {
		return 0;
    }
    
    PCDFileAddress = database_path +"/"+ PCDFileAddress.c_str();

    //load a PCD object  
    boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
    if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *target_pc) == -1)
    {	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
    }
    else
    {
	    ROS_INFO("\t\t[1]-adding a new instance : %s", PCDFileAddress.c_str());
    }
       

    /* ________________________________________________
    |                                                 |
    |  Compute GOOD Description for given point cloud |
    |_________________________________________________| */
  
    boost::shared_ptr<pcl::PointCloud<T> > pca_object_view (new PointCloud<PointT>);
    boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
    vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
    double largest_side = 0;
    int  sign = 1;
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
    ROS_INFO("\nsize of object view histogram %ld",object_representation.spin_image.size());
    
    addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
    return (1);
    
}

int IntroduceNewInstanceDeepLearningUsingGOOD ( string database_path,
												std::string PCDFileAddress,
												unsigned int cat_id, 
												unsigned int track_id,
												unsigned int view_id, 
												int adaptive_support_lenght,
												double global_image_width,
												int threshold,
												int number_of_bins,
												ros::ServiceClient deep_learning_server,
												PrettyPrint &pp )
{

    ROS_INFO ("name of given object view = %s",PCDFileAddress.c_str());
    string categoryName = extractCategoryName(PCDFileAddress);
    
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#' || categoryName == "Category//Unk") // Skip blank lines or comments
    {
		return 0;
    }
    
    PCDFileAddress = database_path +"/"+ PCDFileAddress.c_str();

    //load a PCD object  
    boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
    if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *target_pc) == -1)
    {	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
    }
    else
    {
	    //ROS_INFO("\t\t[1]-IntroduceNewInstance using GOOD descriptor : Loaded a point cloud: %s", PCDFileAddress.c_str());
    }
       

    /* ________________________________________________
    |                                                 |
    |  Compute GOOD Description for given point cloud |
    |_________________________________________________| */
  
    boost::shared_ptr<pcl::PointCloud<T> > pca_object_view (new PointCloud<PointT>);
    boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
    vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
    double largest_side = 0;
    int  sign = 1;
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
    //ROS_INFO("\nsize of object view histogram %ld",object_representation.spin_image.size());
    

	SITOV deep_representation_sitov;
	/// call deep learning service to represent the given GOOD description as vgg16  
	race_deep_learning_feature_extraction::deep_representation srv;
	srv.request.good_representation = object_representation.spin_image;
	if (deep_learning_server.call(srv))
	{
		//pp.info(std::ostringstream().flush() << "################ receive server responce with size of " << srv.response.deep_representation.size() );
		//ROS_INFO("################ receive server responce with size of %ld", srv.response.deep_representation.size() );
		if (srv.response.deep_representation.size() < 1)
			ROS_ERROR("Failed to call deep learning service");
			
		for (size_t i = 0; i < srv.response.deep_representation.size(); i++)
		{
			deep_representation_sitov.spin_image.push_back(srv.response.deep_representation.at(i));
		}
	}
	else
	{
		ROS_ERROR("Failed to call deep learning service");
	}

    //categoryName = extractCategoryName(PCDFileAddress);
    //addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
	//addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, deep_representation_sitov , pp);	
	addObjectViewHistogramInSpecificCategoryDeepLearning(categoryName, 1, track_id, 1, deep_representation_sitov , pp);	
    return (1);
    
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int objectRepresentationBagOfWords (vector <SITOV> cluster_center, 
				    vector <SITOV> object_spin_images, 
				    SITOV  &object_representation )
{
   
    if (object_spin_images.size() == 0 )
    {
	ROS_ERROR("Error: size of object view spin images is zero- could not represent based on BOW");
    }

    for (size_t i = 0; i < cluster_center.size(); i++)
    {
	object_representation.spin_image.push_back(0);
    }
    
    //ROS_INFO("\t\t[-]- size of object view histogram = %ld", object_representation.spin_image.size());
    
    for (size_t i = 0; i < object_spin_images.size(); i++)
    {		
	SITOV sp1;
	sp1=object_spin_images.at(i);
	
	float diffrence=100;
	float diff_temp = 100;
	int id=0;

	for (size_t j = 0; j < cluster_center.size(); j++)
	{
	    SITOV sp2;
	    sp2= cluster_center.at(j);
	    if (!differenceBetweenSpinImage(sp1, sp2, diffrence))
	    {	
		ROS_INFO("\t\t[-]- size of spinimage of cluster center= %ld", sp2.spin_image.size());
		ROS_INFO("\t\t[-]- size of spinimage of the object= %ld" ,sp1.spin_image.size());
		ROS_ERROR("Error comparing spin images");
		return 0;	
	    }
	    //ROS_INFO("\t\t[-]- diffrence[%ld,%ld] = %f    diff_temp =%f",i ,j, diffrence, diff_temp);
	    if ( diffrence < diff_temp)
	    {
		diff_temp = diffrence;
		id = j;
	    } 
	}
	
	//ROS_INFO("\t\t[-]- best_match =%i",id);
	object_representation.spin_image.at(id)++;
    }
    
    //normalizing histogram.
    for (size_t i = 0; i < cluster_center.size(); i++)
    {
		float normalizing_bin = object_representation.spin_image.at(i)/object_spin_images.size();
		object_representation.spin_image.at(i)= normalizing_bin;
    }
    
    
    return (1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int notNormalizedObjectRepresentationBagOfWords (vector <SITOV> cluster_center, 
				    vector <SITOV> object_spin_images, 
				    SITOV  &object_representation )
{
   
    if (object_spin_images.size() == 0 )
    {
	ROS_ERROR("Error: size of object view spin images is zero- could not represent based on BOW");
    }

    for (size_t i = 0; i < cluster_center.size(); i++)
    {
	object_representation.spin_image.push_back(0);
    }
    
    ROS_INFO("\t\t[-]- size of object view histogram = %ld", object_representation.spin_image.size());
    
    for (size_t i = 0; i < object_spin_images.size(); i++)
    {		
	SITOV sp1;
	sp1=object_spin_images.at(i);
	
	float diffrence=100;
	float diff_temp = 100;
	int id=0;

	for (size_t j = 0; j < cluster_center.size(); j++)
	{
	    SITOV sp2;
	    sp2= cluster_center.at(j);
	    if (!differenceBetweenSpinImage(sp1, sp2, diffrence))
	    {	
		ROS_INFO("\t\t[-]- size of spinimage of cluster center= %ld", sp2.spin_image.size());
		ROS_INFO("\t\t[-]- size of spinimage of the object= %ld" ,sp1.spin_image.size());
		ROS_ERROR("Error comparing spin images");
		return 0;	
	    }
	    //ROS_INFO("\t\t[-]- diffrence[%ld,%ld] = %f    diff_temp =%f",i ,j, diffrence, diff_temp);
	    if ( diffrence < diff_temp)
	    {
		diff_temp = diffrence;
		id = j;
	    } 
	}
	
	//ROS_INFO("\t\t[-]- best_match =%i",id);
	object_representation.spin_image.at(id)++;
    }
      
    return (1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////


vector <SITOV>  readClusterCenterFromFile (string path)
{
    std::ifstream clusterCenter(path.c_str());
    std::string clusterTmp;
    vector <SITOV> clusterCenterSpinimages;
    std::setprecision(15);
    while (clusterCenter.good ())// read cluster Center
    { 	
	std::getline (clusterCenter, clusterTmp);
	if(clusterTmp.empty () || clusterTmp.at (0) == '#') // Skip blank lines or comments
	    continue;
	
	string element = "";
	SITOV sp;
	size_t j =0;
	size_t k =0;
	
	//ROS_INFO("\t\t[-]- size = %ld ", clusterTmp.length());
	float value;
	for (size_t i = 0; i < clusterTmp.length(); i++)
	{
	    char ch = clusterTmp[i];
	    //ROS_INFO("\t\t[-]- ch = %c", ch);
	    if (ch != ',')
	    {
		element += ch;
	    }
	    else
	    {
 		   //ROS_INFO("\t\t[-]- element[%ld,%ld] = %s",k ,j, element.c_str());
		   value = atof(element.c_str());
		   //ROS_INFO("\t\t[-]- element[%ld,%ld] = %f",k ,j, value);
		   sp.spin_image.push_back(value);   
		   element = "";    
		   j++;
	    } 
	}
	value = atof(element.c_str());
	sp.spin_image.push_back(value);
	clusterCenterSpinimages.push_back(sp);
	k++;
    }
    
   return (clusterCenterSpinimages);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

int IntroduceNewInstanceHistogram ( std::string PCDFileAddress, 
				    unsigned int cat_id, 
				    unsigned int track_id, 
				    unsigned int view_id,
				    PrettyPrint &pp,
				    int spin_image_width_int,
				    float spin_image_support_lenght_float,
				    size_t subsampled_spin_image_num_keypoints
  				)
{

//     string categoryName = PCDFileAddress;
//     categoryName.resize(13);
    string categoryName = extractCategoryName(PCDFileAddress);
    
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#' || categoryName == "Category//Unk") // Skip blank lines or comments
    {
	return 0;
    }
    
    PCDFileAddress = home_directory_address +"/"+ PCDFileAddress.c_str();

    //load a PCD object  
    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
    if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *PCDFile) == -1)
    {	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
    }
    else
    {
	    ROS_INFO("\t\t[1]-Loaded a point cloud: %s", PCDFileAddress.c_str());
    }
    
    
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
      boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
      objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
      
      boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
      boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
      keypoint_selection( target_pc, 
			    0.01/*uniform_sampling_size*/,
			    uniform_keypoints,
			    uniform_sampling_indices);
	
      ROS_INFO ("number of keypoints = %i", uniform_keypoints->points.size());
      
      
      if (!estimateSpinImages2(target_pc, 
			      0.01 /*downsampling_voxel_size*/, 
			      0.05 /*normal_estimation_radius*/,
			      spin_image_width_int /*spin_image_width*/,
			      0.0 /*spin_image_cos_angle*/,
			      1 /*spin_image_minimum_neighbor_density*/,
			      spin_image_support_lenght_float /*spin_image_support_lenght*/,
			      objectViewSpinImages,
			      uniform_sampling_indices /*subsample spinimages*/))
      {
	   ROS_INFO("Could not compute spin images");
	  return (0);
      }
      

    
    
    //compute Spin Image for given point clould
//     //Call the library function for estimateSpinImages
//     boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
//     objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
// 
//     if (!estimateSpinImages(PCDFile, 
// 			    0.01 /*downsampling_voxel_size*/, 
// 			    0.05 /*normal_estimation_radius*/,
// 			    spin_image_width_int /*spin_image_width*/,
// 			    0.0 /*spin_image_cos_angle*/,
// 			    1 /*spin_image_minimum_neighbor_density*/,
// 			    spin_image_support_lenght_float /*spin_image_support_lenght*/,
// 			    objectViewSpinImages,
// 			    subsampled_spin_image_num_keypoints /*subsample spinimages*/
// 	))
//     {
// 	pp.error(std::ostringstream().flush() << "Could not compute spin images");
//     // 		    pp.printCallback();
// 	return (0);
//     }
//     pp.info(std::ostringstream().flush() << "Computed " << objectViewSpinImages->size() << " spin images for given point cloud. ");

		
    string dictionary_path = ros::package::getPath("race_object_representation") + "/clusters.txt";
    vector <SITOV> cluster_center = readClusterCenterFromFile (dictionary_path);
    
    SITOV object_representation;
    objectRepresentationBagOfWords (cluster_center, *objectViewSpinImages, object_representation);

    //ROS_INFO("size of object view histogram %ld",object_representation.spin_image.size());

    addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);
	

    ROS_INFO("\t\t[-]-%s created...",categoryName.c_str());
        
    return (0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int IntroduceNewInstanceHistogramNotNormalized ( std::string PCDFileAddress, 
						unsigned int cat_id, 
						unsigned int track_id, 
						unsigned int view_id,
						PrettyPrint &pp,
						int spin_image_width_int,
						float spin_image_support_lenght_float,
						size_t subsampled_spin_image_num_keypoints						 
						)
{

//     string categoryName = PCDFileAddress;
//     categoryName.resize(13);
    string categoryName = extractCategoryName(PCDFileAddress);
    
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#' || categoryName == "Category//Unk") // Skip blank lines or comments
    {
	return 0;
    }
    
    PCDFileAddress = home_directory_address +"/"+ PCDFileAddress.c_str();

    //load a PCD object  
    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
    if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *PCDFile) == -1)
    {	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
    }
    else
    {
	    ROS_INFO("\t\t[1]-Loaded a point cloud: %s", PCDFileAddress.c_str());
    }
    
    //compute Spin Image for given point clould
    
    //Call the library function for estimateSpinImages
    boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
    objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);

    if (!estimateSpinImages(PCDFile, 
			    0.01 /*downsampling_voxel_size*/, 
			    0.05 /*normal_estimation_radius*/,
			    spin_image_width_int /*spin_image_width*/,
			    0.0 /*spin_image_cos_angle*/,
			    1 /*spin_image_minimum_neighbor_density*/,
			    spin_image_support_lenght_float /*spin_image_support_lenght*/,
			    objectViewSpinImages,
			    subsampled_spin_image_num_keypoints /*subsample spinimages*/
	))
    {
	pp.error(std::ostringstream().flush() << "Could not compute spin images");
    // 		    pp.printCallback();
	return (0);
    }
    pp.info(std::ostringstream().flush() << "Computed " << objectViewSpinImages->size() << " spin images for given point cloud. ");

			    

    
    	string dictionary_path = ros::package::getPath("race_object_representation") + "/clusters.txt";
	vector <SITOV> cluster_center = readClusterCenterFromFile (dictionary_path);
	
	SITOV object_representation;
	notNormalizedObjectRepresentationBagOfWords (cluster_center, *objectViewSpinImages, object_representation);

	//ROS_INFO("size of object view histogram %ld",object_representation.spin_image.size());

	addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);
	

    ROS_INFO("\t\t[-]-%s created...",categoryName.c_str());
        
    return (0);
}



////////////////////////////////////////////////////////////////////////////////////////////////////
int IntroduceNewInstanceHistogramNotNormalized2 ( std::string PCDFileAddress, 
						unsigned int cat_id, 
						unsigned int track_id, 
						unsigned int view_id,
						PrettyPrint &pp,
						int spin_image_width_int,
						float spin_image_support_lenght_float,
						float uniform_sampling_size,
						vector <SITOV> dictionary
						)
{

//     string categoryName = PCDFileAddress;
//     categoryName.resize(13);
    string categoryName = extractCategoryName(PCDFileAddress);
    
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#' || categoryName == "Category//Unk") // Skip blank lines or comments
    {
	return 0;
    }
    
    PCDFileAddress = home_directory_address +"/"+ PCDFileAddress.c_str();

    //load a PCD object  
    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
    if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *PCDFile) == -1)
    {	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
    }
    else
    {
	    ROS_INFO("\t\t[1]-Loaded a point cloud: %s", PCDFileAddress.c_str());
    }
    
    /* ________________________________________________
    |                                                 |
    |  Compute the Spin-Images for given point cloud  |
    |_________________________________________________| */
    
    //Declare a boost share ptr to the spin image msg
    boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
    objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
    
    boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
    boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
    keypoint_selection( PCDFile, 
			uniform_sampling_size,
			uniform_keypoints,
			uniform_sampling_indices);
    
    ROS_INFO ("uniform_sampling_size = %f", uniform_sampling_size);
    ROS_INFO ("number of keypoints = %i", uniform_keypoints->points.size());
    
    
    if (!estimateSpinImages2(PCDFile, 
			    0.01 /*downsampling_voxel_size*/, 
			    0.05 /*normal_estimation_radius*/,
			    spin_image_width_int /*spin_image_width*/,
			    0.0 /*spin_image_cos_angle*/,
			    1 /*spin_image_minimum_neighbor_density*/,
			    spin_image_support_lenght_float /*spin_image_support_lenght*/,
			    objectViewSpinImages,
			    uniform_sampling_indices /*subsample spinimages*/
	))
    {
	pp.error(std::ostringstream().flush() << "Could not compute spin images");
	return (0);
    }
    pp.info(std::ostringstream().flush() << "Computed " << objectViewSpinImages->size() << " spin images for given point cloud. ");
    
    //string dictionary_path = ros::package::getPath("race_object_representation") + "/clusters.txt";
    //vector <SITOV> cluster_center = readClusterCenterFromFile (dictionary_path);

    SITOV object_representation;
    notNormalizedObjectRepresentationBagOfWords (dictionary, *objectViewSpinImages, object_representation);

    //ROS_INFO("size of object view histogram %ld",object_representation.spin_image.size());

    addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);
	

    ROS_INFO("\t\t[-]-%s created...",categoryName.c_str());
        
    return (0);
}


int NumberofCategoriesinDataset (string home_address)
{   
    //Dataset Path
    string path = home_address+ "Category/Category_orginal.txt";
    ROS_INFO("\t\t[-]- database path = %s", path.c_str());
    std::ifstream listOfObjectCategories (path.c_str(), std::ifstream::in);

    int number_of_categories =0;
    size_t total_number_of_instances = 0;

    while(listOfObjectCategories.good ())
    {
	string categoryAddress;
	std::getline (listOfObjectCategories, categoryAddress);
	if(categoryAddress.empty () || categoryAddress.at (0) == '#') // Skip blank lines or comments
	{
	    continue;
	}
	number_of_categories ++;
    }
    
    return (number_of_categories);
}

float contextChangeProbability (int category_number, int number_of_categories_in_dataset)
{
   float p = pow (number_of_categories_in_dataset, -0.5) * pow(category_number, 0.5);
   return (p);
}

bool inContext(int category_index, int context_index, 
	       int number_of_contexts, int overlap_between_contexts, int number_of_categories_in_dataset )
{
      int start_index = int(number_of_categories_in_dataset/number_of_contexts) * (context_index);
      int end_index = int(number_of_categories_in_dataset/number_of_contexts) * (context_index+1) + overlap_between_contexts ;    
      if ((category_index >= start_index) && (category_index < end_index))
      { 
	  return (true);
      }
      else 
      {
	  return (false);	
      }
}

bool inContext(int category_index, int context_index, int context_change_index, int number_of_categories_in_dataset )
{
      int start_index = context_change_index * (context_index);
      int end_index = context_change_index ;
      if (context_index == 1)
      {
	end_index = number_of_categories_in_dataset ;
      }   
      if ((category_index > start_index) && (category_index <= end_index))
      { 
	  return (true);
      }
      else 
      {
	  return (false);	
      }
}



int selectAnInstancefromSpecificCategory(unsigned int category_index, 
					 unsigned int &instance_number, 
					 string &Instance)
{
    std::string path;
    path =  home_directory_address +"/Category/Category.txt";
    //ROS_INFO("path = %s",path.c_str());
    //ROS_INFO("TEST");
    ROS_INFO("\t\t[-] category index = %d",category_index);
    ROS_INFO("\t\t[-] instance_number = %d",instance_number);
//     
    std::ifstream listOfObjectCategoriesAddress (path.c_str());
    std::string categoryAddresstmp;
    std::string categoryName;

    unsigned int cat_index = 0;
    while ((listOfObjectCategoriesAddress.good()) && (cat_index < category_index))
    {
		std::getline (listOfObjectCategoriesAddress, categoryAddresstmp);
		if(categoryAddresstmp.empty () || categoryAddresstmp.at (0) == '#') // Skip blank lines or comments
			continue;

		if (cat_index < category_index)
			cat_index++;
    }
    
    if (cat_index != category_index)
    {
		ROS_INFO("\t\t[-]-The file doesn't exist - condition : cat_index != category_index, cat_index = %i", cat_index);
		return -1;
    }
    
    path = home_directory_address +"/"+ categoryAddresstmp.c_str();
    
    std::ifstream categoyInstances (path.c_str());
    std::string PCDFileAddressTmp;
    
    unsigned int inst_number =0;
    while ((categoyInstances.good ()) && (inst_number < instance_number))// read instances of a category 
    {	
		std::getline (categoyInstances, PCDFileAddressTmp);
		if(PCDFileAddressTmp.empty () || PCDFileAddressTmp.at (0) == '#') // Skip blank lines or comments
			continue;
		if (inst_number < instance_number)
	    inst_number ++;
    }
    if (inst_number < instance_number)
    {
		ROS_INFO("\t\t[-]- category path= %s", path);
		ROS_INFO("\t\t[-]-The file doesn't exist - condition : inst_number < instance_number -- inst_number = %i", inst_number);
		return -1;
    }
    
    Instance=PCDFileAddressTmp;
    instance_number++;
    return 0;
    
}


vector <int> generateSequence (int n)
{
    /* initialize random seed: */
    srand (time(NULL));
    vector <int> sequence;
    while( sequence.size() < n )
    {    
	/* generate random number between 1 and 10: */
	int num = rand() % n + 1;
	bool falg = false;
	for (int j= 0; j < sequence.size(); j++)
	{
	    if (num == sequence.at(j))
	    {
		falg= true;
		break;
	    }
	}
	if (falg == false)
	{
	   sequence.push_back(num);
	}
	//ROS_INFO("\n size= %i \n",sequence.size()); 
    }
    
    return(sequence);
 // test :
//     vector <int> test;
//     test = generateSequence(15);
//     for (int i = 0; i< test.size(); i++)
//     {
// 		printf(" %i,",test.at(i));
//     }
}

int generateRrandomSequencesInstances (string path)
{    
    string path1= home_directory_address + path;    
 //   ROS_INFO("\n path-instance_original= %s \n",path1.c_str());

    std::ifstream listOfObjectInstancesAddress (path1.c_str());
    string InstanceAddresstmp = "";
    unsigned int number_of_exist_instances = 0;
    while (listOfObjectInstancesAddress.good()) 
    {
	std::getline (listOfObjectInstancesAddress, InstanceAddresstmp);
	if(InstanceAddresstmp.empty () || InstanceAddresstmp.at (0) == '#') // Skip blank lines or comments
	    continue;
	number_of_exist_instances++;
    }
   // ROS_INFO("\n number_of_exist_instances= %i \n",number_of_exist_instances);
    
    InstanceAddresstmp = "";
    vector <int> instances_sequence = generateSequence (number_of_exist_instances);
    std::ofstream instances;
    string path2 = home_directory_address + path;
    path2.resize(path2.size()-12);
    path2+= ".txt";
    //ROS_INFO("\n reorder category = %s \n",path2.c_str());

    instances.open (path2.c_str(), std::ofstream::out);
    for (int i =0; i < instances_sequence.size(); i++)
    {
	std::ifstream listOfObjectCategories (path1.c_str());
	int j = 0;
	while ((listOfObjectCategories.good()) && (j < instances_sequence.at(i)))
	{
	    std::getline (listOfObjectCategories, InstanceAddresstmp);
	    j++;
	}
	instances << InstanceAddresstmp.c_str()<<"\n";
	//ROS_INFO("\n instance= %s \n",InstanceAddresstmp.c_str());
    }
    instances.close();
    
    return (0);
}

int generateRrandomSequencesCategories (int RunCount)
{

    std::string path;
    ROS_INFO ("home_directory_address = %s", home_directory_address.c_str());

    path = home_directory_address +"/Category/Category_orginal.txt";
    
    std::ifstream listOfObjectCategoriesAddress (path.c_str());
    ROS_INFO ("path = %s", path.c_str());

    string categoryAddresstmp = "";
    unsigned int number_of_exist_categories = 0;
    while (listOfObjectCategoriesAddress.good()) 
    {
	std::getline (listOfObjectCategoriesAddress, categoryAddresstmp);
	if(categoryAddresstmp.empty () || categoryAddresstmp.at (0) == '#') // Skip blank lines or comments
	    continue;
	number_of_exist_categories++;
    }
        	
        	
    ROS_INFO ("number of category = %i", number_of_exist_categories);

    vector <int> categories_sequence = generateSequence (number_of_exist_categories);

//     for (int i =0; i< categories_sequence.size();i++)
//     {
// 	ROS_INFO("S [%i]= %i",i, categories_sequence.at(i));
//     }
    
    std::ofstream categoies;
    string path2 = home_directory_address +"/Category/Category.txt";
    categoies.open (path2.c_str(), std::ofstream::out);
    for (int i =0; i < categories_sequence.size(); i++)
    {
	std::ifstream listOfObjectCategories (path.c_str());
	int j = 0;
	while ((listOfObjectCategories.good()) && (j < categories_sequence.at(i)))
	{
	    std::getline (listOfObjectCategories, categoryAddresstmp);
	    generateRrandomSequencesInstances(categoryAddresstmp.c_str());
	    j++;
	}
	categoryAddresstmp.resize(categoryAddresstmp.size()-12);
	categoryAddresstmp+= ".txt";
	categoies << categoryAddresstmp.c_str()<<"\n";
    }
    return 0 ;
  
}

float compute_precision_of_last_3n (vector <int> recognition_results ,
				     int number_of_taught_categories)
{
    ROS_INFO("\t\t[-]- ^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&");
    ROS_INFO("\t\t[-]- Inside compute_precision_of_last_3n function");
    ROS_INFO("\t\t[-]- size of recognition result = %d",recognition_results.size() );
    if (recognition_results.size() <1)
    {
	printf(" Error: size of recognition result array is %i",recognition_results.size() );
	return 0;
    }
    for (int i =0; i< recognition_results.size(); i++)
    {
	    printf("%d, ",recognition_results.at(i) );
    }
    printf("\n");
    
    float precision = 0;
    int TP=0; int FP=0; int FN=0;
    
    int win_size = 3*number_of_taught_categories;
    ROS_INFO("\ni start from %d till %d",recognition_results.size() - win_size -1, recognition_results.size()-1 );
    ROS_INFO("windows size = %d",win_size );
    ROS_INFO("number_of_taught_categories = %d",number_of_taught_categories );

    
    for (int i = recognition_results.size()-1; i > recognition_results.size() - win_size ; i--)
    {
	int result = recognition_results.at(i);
// 	printf("%d, ",result );
	if (result==1)
	{
	    TP++;
	}
	else if (result==2)
	{
	    FP++;
	}
	else if (result==3)
	{
	    FN++;
	}
	else if (result==4)
	{
	    FP++;FN++;
	}
    }
    
    ROS_INFO("\t\t[-]- number of TP= %d, FP= %d, FN=%d", TP,FP,FN); 
    float Precision = TP/double (TP+FP);
    ROS_INFO("\t\t[-]- Precision = %f", Precision); 
    
    
//     cout << "\nTP =" <<TP;
//     cout << "\nFP =" <<FP;
//     cout << "\nFN =" <<FN;		
//     char ch;
//     cin >>ch;
    
    return(Precision);
}


float computeF1OfLast3n (vector <int> recognition_results ,
				     int number_of_taught_categories)
{
    ROS_INFO("\t\t[-]- ^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&");
    ROS_INFO("\t\t[-]- Inside compute_precision_of_last_3n function");
    ROS_INFO("\t\t[-]- size of recognition result = %d",recognition_results.size() );
    if (recognition_results.size() <1)
    {
	printf(" Error: size of recognition result array is %i",recognition_results.size() );
	return 0;
    }
    for (int i =0; i< recognition_results.size(); i++)
    {
	    printf("%d, ",recognition_results.at(i) );
    }
    printf("\n");
    
    float precision = 0;
    int TP = 0; int FP = 0; int FN = 0;
    
    int win_size = 3*number_of_taught_categories;
    ROS_INFO("\ni start from %d till %d",recognition_results.size() - win_size -1, recognition_results.size()-1 );
    ROS_INFO("windows size = %d",win_size );
    ROS_INFO("number_of_taught_categories = %d",number_of_taught_categories );

    
    for (int i = recognition_results.size()-1; i > recognition_results.size() - win_size ; i--)
    {
	int result = recognition_results.at(i);
// 	printf("%d, ",result );
	if (result==1)
	{
	    TP++;
	}
	else if (result==2)
	{
	    FP++;
	}
	else if (result==3)
	{
	    FN++;
	}
	else if (result==4)
	{
	    FP++;FN++;
	}
    }
    
    
    
    ROS_INFO("\t\t[-]- number of TP= %d, FP= %d, FN=%d", TP,FP,FN); 
    
    
    float Precision = TP/double (TP+FP);
    float Recall = TP/double (TP+FN);    
    float F1 = 2 * (Precision * Recall )/(Precision + Recall );
    ROS_INFO("\t\t[-]- F1 = %f", F1); 
    
    
//     cout << "\nTP =" <<TP;
//     cout << "\nFP =" <<FP;
//     cout << "\nFN =" <<FN;		
//     char ch;
//     cin >>ch;
    
    return(F1);
}



int compute_Precision_Recall_Fmeasure_of_last_3n (vector <int> recognition_results ,
				     int number_of_taught_categories, 
				     float &Precision, float &Recall, float &F1)
{
//     ROS_INFO("\t\t[-]- ^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&^&");
//     ROS_INFO("\t\t[-]- Inside compute_precision_of_last_3n function");
//     ROS_INFO("\t\t[-]- size of recognition result = %d",recognition_results.size() );
    if (recognition_results.size() <1)
    {
	printf(" Error: size of recognition result array is %i",recognition_results.size() );
	return 0;
    }
    for (int i =0; i< recognition_results.size(); i++)
    {
// 	    printf("%d, ",recognition_results.at(i) );
    }
//     printf("\n");
    
    float precision = 0;
    int TP=0; int FP=0; int FN=0;
    
    int win_size = 3*number_of_taught_categories;
//     ROS_INFO("\ni start from %d till %d",recognition_results.size() - win_size -1, recognition_results.size()-1 );
//     ROS_INFO("windows size = %d",win_size );
//     ROS_INFO("number_of_taught_categories = %d",number_of_taught_categories );

    
    for (int i = recognition_results.size()-1; i > recognition_results.size() - win_size ; i--)
    {
	int result = recognition_results.at(i);
// 	printf("%d, ",result );
	if (result==1)
	{
	    TP++;
	}
	else if (result==2)
	{
	    FP++;
	}
	else if (result==3)
	{
	    FN++;
	}
	else if (result==4)
	{
	    FP++;FN++;
	}
    }
    
    
    
//     ROS_INFO("\t\t[-]- number of TP= %d, FP= %d, FN=%d", TP,FP,FN); 
    
    
    Precision = TP/double (TP+FP);
    Recall = TP/double (TP+FN);    
    F1 = 2 * (Precision * Recall )/(Precision + Recall );
//     ROS_INFO("\t\t[-]- F1 = %f", F1); 
    
    
//     cout << "\nTP =" <<TP;
//     cout << "\nFP =" <<FP;
//     cout << "\nFN =" <<FN;		
//     char ch;
//     cin >>ch;
    
    return(F1);
}





void monitorPrecision (string precision_file, float Precision )
{
    std::ofstream PrecisionMonitor;
    PrecisionMonitor.open (precision_file.c_str(), std::ofstream::app);
    PrecisionMonitor.precision(4);
    PrecisionMonitor << Precision<<"\n";
    PrecisionMonitor.close();
    
}

void monitorF1VsLearnedCategory (string f1_learned_caegory, int TP, int FP, int FN )
{
    double Precision, Recall, F1;
    if ((TP+FP)!=0)
    {
	Precision = TP/double (TP+FP);
    }
    else
    {
	Precision = 0;
    }		
    if ((TP+FN)!=0)
    {
	Recall = TP/double (TP+FN);
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
  
    std::ofstream f1_vs_learned_caegory;
    f1_vs_learned_caegory.open (f1_learned_caegory.c_str(), std::ofstream::app);
    f1_vs_learned_caegory.precision(4);
    f1_vs_learned_caegory << F1<<"\n";
    f1_vs_learned_caegory.close();
    
}

void reportCurrentResults(int TP, int FP, int FN, string fname, bool global)
{
	double Precision = TP/double (TP+FP);
	double Recall = TP/double (TP+FN);
	
	std::ofstream Result_file;
	Result_file.open (fname.c_str(), std::ofstream::app);
	Result_file.precision(4);
	if (global)
		Result_file << "\n\n\t******************* Global *********************";
	else	Result_file << "\n\n\t******************* Lastest run ****************";
	Result_file << "\n\t\t - True  Positive = "<< TP;
	Result_file << "\n\t\t - False Positive = "<< FP;
	Result_file << "\n\t\t - False Negative = "<< FN;
	Result_file << "\n\t\t - Precision  = "<< Precision;
	Result_file << "\n\t\t - Recall = "<< Recall;
	Result_file << "\n\t\t - F1 = "<< 2 * (Precision * Recall )/(Precision + Recall );
	Result_file << "\n\n\t************************************************\n\n";
	
	// Result << "\n\n\t***********average_class_precision**************\n\n";
	// Result << "\n\t\t - average_class_precision = "<< average_class_precision_value;
	
	Result_file << "\n------------------------------------------------------------------------------------------------------------------------------------";
	Result_file.close();
}

void report_category_introduced(string fname, string cat_name)
{
	std::ofstream Result_file;
	Result_file.open (fname.c_str(), std::ofstream::app);
	Result_file << "\n\t********************************************";
	Result_file << "\n\t\t - "<< cat_name.c_str() <<" Category Introduced";
	Result_file << "\n\t********************************************";
	Result_file.close();
}

void report_precision_of_last_3n(string fname, double Precision)
{
	std::ofstream Result_file;
	Result_file.open (fname.c_str(), std::ofstream::app);
	Result_file << "\n\t*********** Precision of last 3n **************";
	Result_file << "\n\t\t - precision = "<< Precision;
	Result_file << "\n\t********************************************";
	Result_file.close();
}


void reportF1OfLast3n(string fname, double F1)
{
	std::ofstream Result_file;
	Result_file.open (fname.c_str(), std::ofstream::app);
	Result_file << "\n\t*********** F1 of last 3n **************";
	Result_file << "\n\t\t - F1 = "<< F1;
	Result_file << "\n\t********************************************";
	Result_file.close();
}

int reportExperimentResult (vector <float> average_class_precision,
			     int number_of_stored_instances, 
			     int number_of_taught_categories,  
			     string fname, ros::Duration duration)
{
   
    float average_class_precision_value =0;
    for (int i =0; i<average_class_precision.size(); i++)
    {
	average_class_precision_value+=average_class_precision.at(i); 	    
    }
    average_class_precision_value=average_class_precision_value/average_class_precision.size();
    double duration_sec = duration.toSec();
    
    std::ofstream Result;
    Result.open (fname.c_str(), std::ofstream::app);
    Result.precision(4);
//     Result << "\n\t -Note: the experiment is terminated because there is not\n\t\tenough test data to continue the evaluation\n";
    Result << "\n\n\t************** Expriment Result ****************";
    Result << "\n\t\t - Average_class_precision = "<< average_class_precision_value;
    Result << "\n\t\t - All stored instances = "<< number_of_stored_instances;
    Result << "\n\t\t - Number of taught categories = "<< number_of_taught_categories;
    Result << "\n\t\t - Average number of instances per category = "<< float (number_of_stored_instances) / float (number_of_taught_categories);	
    Result << "\n\t\t - This expriment took " << duration_sec << " secs";
    Result << "\n\n\t************************************************\n\n";
    Result.close();
    return (0);
}


int introduceNewCategory(int class_index,
			  unsigned int &track_id,
			  unsigned int &instance_number,
			  string fname, 
			  int spin_image_width_int,
			  float spin_image_support_lenght_float,
			  size_t subsampled_spin_image_num_keypoints
 			)
{
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
// 	    selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path);
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
			ROS_ERROR("\t\t[-] The object view or category does not exist");
			return -1;// ERROR : file doesn't exist
	    }
	    //ROS_INFO("\t\t[-]-Instance number %d", instance_number);
	    //ROS_INFO("\t\t[-] Instance path %s", instance_path.c_str());
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
	    IntroduceNewInstance(instance_path, 
				 cat_id, track_id, view_id,
				 spin_image_width_int,
				 spin_image_support_lenght_float,
				 subsampled_spin_image_num_keypoints
				); 
	    track_id++;
	    //view_id ++; // in this implementation we consider VID as a constant
	}
	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	//ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 
 	report_category_introduced(fname,categoryName.c_str());
	return 0;

    
}

int introduceNewCategoryDeepLearningUsingGOOD(string home_address,
												int class_index,
												unsigned int &track_id,
												unsigned int &instance_number,
												string fname, 
												int adaptive_support_lenght,
												double global_image_width,
												int threshold,
												int number_of_bins, 
												ros::ServiceClient deep_learning_server
												)
{
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
// 	    selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path);
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
			ROS_ERROR("\t\t[-]-The object view or category does not exist");
			//i -=1;
			//continue;
			return -1;// ERROR : file doesn't exist
	    }
	    //ROS_INFO("\t\t[-] Instance path %s", instance_path.c_str());
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
	      
	    PrettyPrint pp;
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
	    //view_id ++; // in this implementation we consider VID as a constant
	}
	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	//ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 
 	//report_category_introduced(fname,categoryName.c_str());
	return 0;

    
}

int introduceNewCategoryGOOD(string home_address,
			  int class_index,
			  unsigned int &track_id,
			  unsigned int &instance_number,
			  string fname, 
			  int adaptive_support_lenght,
			  double global_image_width,
			  int threshold,
			  int number_of_bins
 			)
{
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
			ROS_ERROR("\t\t[-]-The object view or category does not exist");
			return -1;// ERROR : file doesn't exist
	    }
	    ROS_INFO("\t\t[-]-Instance path = %s", instance_path.c_str());
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
	      
	    PrettyPrint pp;
	    introduceNewInstanceGOOD ( home_address,
					instance_path,
					cat_id, track_id, view_id,
					adaptive_support_lenght,
					global_image_width,
					threshold,
					number_of_bins,
					pp );
	    
	    track_id++;
	}
	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 
 	//report_category_introduced(fname,categoryName.c_str());
	return 0;

    
}

int introduceNewCategoryGOODAndUpdatingNaiveBayesModel(string home_address,
			  int class_index,
			  unsigned int &track_id,
			  unsigned int &instance_number,
			  string fname, 
			  int adaptive_support_lenght,
			  double global_image_width,
			  int threshold,
			  int number_of_bins
 			)
{

	PrettyPrint pp;
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
// 	    selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path);
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
		ROS_ERROR("\t\t[-]-The object view or category does not exist");
		return -1;// ERROR : file doesn't exist
	    }
	    ROS_INFO("\t\t[-]-Instance path %s", instance_path.c_str());
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
	      
	    introduceNewInstanceGOOD ( home_address,
					instance_path,
					cat_id, track_id, view_id,
					adaptive_support_lenght,
					global_image_width,
					threshold,
					number_of_bins,
					pp );
	    
	    track_id++;
	    //view_id ++; // in this implementation we consider VID as a constant
	}
	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 
 	//report_category_introduced(fname,categoryName.c_str());

	updateGOODNaiveBayesModel(categoryName, 1, pp);
	ROS_INFO("\n Naive Bayes Models updated... %s", categoryName.c_str()); 
	report_category_introduced(fname,categoryName.c_str());

	ros::spinOnce();
	return 0;
    
}





int introduceNewCategory2(string home_address,
			  int class_index,
			  unsigned int &track_id,
			  unsigned int &instance_number,
			  string fname, 
			  int spin_image_width_int,
			  float spin_image_support_lenght_float,
			  float uniform_sampling_size
 			)
{
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
// 	    selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path);
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
		ROS_ERROR("\t\t[-]-The object view or category does not exist");
		return -1;// ERROR : file doesn't exist
	    }
	    ROS_INFO("\t\t[-]-Instance path %s", instance_path.c_str());
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
	      
	    IntroduceNewInstance2(home_address,
				  instance_path, 
				  cat_id, track_id, view_id,
				  spin_image_width_int,
				  spin_image_support_lenght_float,
				  uniform_sampling_size ); 
	    track_id++;
	    //view_id ++; // in this implementation we consider VID as a constant
	}
	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 
 	//report_category_introduced(fname,categoryName.c_str());
	return 0;

    
}


int introduceNewCategoryHistogram(int class_index,
				   unsigned int &track_id,
				   unsigned int &instance_number,
				   string fname,
				   PrettyPrint &pp,
				   int spin_image_width_int,
				   float spin_image_support_lenght_float,
				   size_t subsampled_spin_image_num_keypoints
  				)
				  
{
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
	    ROS_INFO("\t\t[-]-inside introduceNewCategoryHistogram function Instance number %d", instance_number);
// 	    selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path);
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
		ROS_ERROR("\t\t[-]-The object view or category does not exist");
		return -1;// ERROR : file doesn't exist
	    }
	    
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
// 	    IntroduceNewInstance(instance_path, cat_id, track_id, view_id); 
	    IntroduceNewInstanceHistogram ( instance_path, 
					    cat_id, 
					    track_id, 
					    view_id,
					    pp,
					    spin_image_width_int,
					    spin_image_support_lenght_float,
					    subsampled_spin_image_num_keypoints
  					);
	    track_id++;
	    //view_id ++; // in this implementation we consider VID as a constant
	}

	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 
	report_category_introduced(fname,categoryName.c_str());
	return 0;
}


int introduceNewCategoryHistogramAndUpdatingNaiveBayesModel(int class_index,
							    unsigned int &track_id,
							    unsigned int &instance_number,
							    string fname,
							    PrettyPrint &pp,
							    int spin_image_width_int,
							    float spin_image_support_lenght_float,
							    size_t subsampled_spin_image_num_keypoints
							    )
				  
{
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
	    ROS_INFO("\t\t[-]- inside introduceNewCategoryHistogramAndUpdatingNaiveBayesModel function Instance number %d", instance_number);
// 	    selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path);
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
		ROS_ERROR("\t\t[-]-The object view or category does not exist");
		return -1;// ERROR : file doesn't exist
	    }
	    
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
	    IntroduceNewInstanceHistogramNotNormalized ( instance_path, 
							 cat_id, 
							 track_id, 
							 view_id,
							 pp,
							 spin_image_width_int,
							 spin_image_support_lenght_float,
							 subsampled_spin_image_num_keypoints 
   						    );
	    track_id++;
	    //view_id ++; // in this implementation we consider VID as a constant
	}

	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 

	updateNaiveBayesModel(categoryName, 1, pp);
	ROS_INFO("\n Naive Bayes Models updated... %s", categoryName.c_str()); 
	report_category_introduced(fname,categoryName.c_str());

	ros::spinOnce();
}

int introduceNewCategoryHistogramAndUpdatingNaiveBayesModel2(int class_index,
							    unsigned int &track_id,
							    unsigned int &instance_number,
							    string fname,
							    PrettyPrint &pp,
							    int spin_image_width_int,
							    float spin_image_support_lenght_float,
							    float uniform_sampling_size,
							    vector <SITOV> dictionary
							    )
				  
{
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
	    ROS_INFO("\t\t[-]- inside introduceNewCategoryHistogramAndUpdatingNaiveBayesModel function Instance number %d", instance_number);
// 	    selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path);
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
		ROS_ERROR("\t\t[-]-The object view or category does not exist");
		return -1;// ERROR : file doesn't exist
	    }
	    
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
	    IntroduceNewInstanceHistogramNotNormalized2 ( instance_path, 
							 cat_id, 
							 track_id, 
							 view_id,
							 pp,
							 spin_image_width_int,
							 spin_image_support_lenght_float,
							 uniform_sampling_size,
							 dictionary  );
	    track_id++;
	    //view_id ++; // in this implementation we consider VID as a constant
	}

	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 

	updateNaiveBayesModel(categoryName, 1, pp);
	ROS_INFO("\n Naive Bayes Models updated... %s", categoryName.c_str()); 
	report_category_introduced(fname,categoryName.c_str());

	ros::spinOnce();
}


void plotSimulatedTeacherProgressInMatlab( int RunCount, float P_Threshold, string precision_file)
{
 
    // If the text inside the plot is not well-appear, please install the font in ubuntu using the following commands.
    // sudo apt-get install xfonts-75dpi
    // sudo apt-get install xfonts-75dpi
    
    char run_count [10];
	vector<float> acc;

    sprintf( run_count, "%d",RunCount );
    string path = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/graph1.m";
        
    ROS_INFO("\t\tpath = %s", path.c_str());

    ofstream matlabFile;
    matlabFile.open (path.c_str(), std::ofstream::out);
    matlabFile.precision(4);
    matlabFile << "close all;\nfigure;\nhold on;\ngrid on;";
    matlabFile << "\nset(gca,'LineStyleOrder', '-');";
    matlabFile << "\nPrecision= [";
    
    std::string value;
    std::ifstream ReadPrecisionMonitor (precision_file.c_str());
    std::getline (ReadPrecisionMonitor, value);
    matlabFile << value;
	acc.push_back(atof (value.c_str()));

    int iteration = 1;
    while (ReadPrecisionMonitor.good())
    {
		std::getline (ReadPrecisionMonitor, value);
		matlabFile << ", "<< value;
		acc.push_back(atof (value.c_str()));
		iteration++;
    }
    
    int remain = iteration % 10;
    matlabFile << "];\naxis([0,"<< iteration + 10 - remain <<",0,1.2]);\nplot (Precision, 'LineWidth',2);";
    matlabFile << "\nxlabel('Question / Correction Iterations','FontSize',15);\nylabel('Protocol Accuracy','FontSize',15);";

    matlabFile << "\nset(gca,'LineStyleOrder', '--');";
    //draw a threshold line
    matlabFile << "\nline([0 "<< iteration + 10 - remain <<"],["<< P_Threshold << " " <<P_Threshold <<"] ,'Color',[0 0 0 0.3], 'LineWidth',1);";
    matlabFile << "\ntext(" << iteration - remain - 10 << "," << P_Threshold + 0.05 <<",'Thereshold', 'FontSize',12, 'Interpreter','Latex');";

    iteration = 0;
    string path_tmp = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/Category_Introduced.txt";
    std::ifstream read_category_introduced (path_tmp.c_str());

    string path_tmp2 = home_directory_address+ "/Category/Category.txt";
    std::ifstream category_name (path_tmp2.c_str());
    string cat_name ="";
    
    value = "";
    string value_tmp;
    // draw a line 
    std::getline (read_category_introduced, value_tmp);
    matlabFile << "\nline(["<< iteration<< ","<< iteration <<"] ,[0.05,1],'Color',[1 0 0], 'LineWidth',1);";
    
    // add category name to the graph
    std::getline (category_name, cat_name);
    cat_name = extractCategoryName(cat_name);
    
    int lfind =  cat_name.find("_");
    if (lfind > 0) 
		cat_name.replace(lfind, 1,1, '-');


    float text_Y_pos = 0.05;
    bool flg=true;
    matlabFile << "\ntext("<<iteration<<".5, 0.05 ,'"<< cat_name <<"' , 'FontSize',15,'FontWeight','bold','Interpreter','Latex' ,'BackgroundColor', [1,0,0,0.2]);";
        
    std::getline (read_category_introduced, value_tmp);
    matlabFile << "\nline(["<< iteration<< ","<< iteration <<"] ,[0.1,1],'Color',[1 0 0], 'LineWidth',1);";	    
    std::getline (category_name, cat_name);
    
    cat_name = extractCategoryName(cat_name);
    lfind =  cat_name.find("_");
    if (lfind > 0) 
	cat_name.replace(lfind, 1,1, '-');
    
    matlabFile << "\ntext("<<iteration<<".5, 0.1,'"<< cat_name <<"' , 'FontSize',15,'FontWeight','bold','Interpreter','Latex', 'BackgroundColor', [1,0,0,0.2] );";

    iteration = 1;
    float y = 0.15;
    while (read_category_introduced.good())
    {

		std::getline (read_category_introduced, value);
		if (strcmp(value.c_str(),"1") == 0)
		{
			std::getline (category_name, cat_name);

			cat_name = extractCategoryName(cat_name);
			lfind = cat_name.find("_");
			if (lfind > 0) 
			{
				cat_name.replace(lfind, 1,1, '-');
			}

			if (acc.at(iteration-1) == 0)
			{
				matlabFile << "\nline([" << iteration << "," << iteration <<"] ,["<< 1 - y <<", 0],'Color',[1 0 0], 'LineWidth',1);";	    
				matlabFile << "\ntext(" << iteration <<".5," << 1 - y <<" ,'"<< cat_name <<"' , 'FontSize',15,'FontWeight','bold','Interpreter','Latex', 'BackgroundColor', [1,0,0,0.2]);";
			}
			else
			{
				matlabFile << "\nline(["<< iteration<< ","<< iteration <<"] ,["<< y <<", 1],'Color',[1 0 0], 'LineWidth',1);";	    
				matlabFile << "\ntext("<<iteration<<".5,"<< y <<" ,'"<< cat_name <<"' , 'FontSize',15,'FontWeight','bold','Interpreter','Latex', 'BackgroundColor', [1,0,0,0.2]);";
			}
			if (iteration == 1) 
				iteration ++ ;
			


			if (y <=0.55)
			{
				y+=0.05;
			}
			else
			{
				y=0.05;
			}
		}
		else
		{
			iteration++;
		}
		
    }
    matlabFile.close();
    ROS_INFO("\t\tMATLAB file created...");
}


void plotLocalF1VsNumberOfLearnedCategoriesInMatlab( int RunCount, 
													 float P_Threshold, 
													 string local_F1_vs_learned_category)
{
    // If the text inside the plot is not well-appear, please install the font in ubuntu using the following commands.
    // sudo apt-get install xfonts-75dpi
    
    char run_count [10];
    sprintf( run_count, "%d",RunCount );
    string path = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/local_F1_vs_learned_category.m";
        
    ROS_INFO("\t\tpath = %s", path.c_str());

    ofstream matlabFile;
    matlabFile.open (path.c_str(), std::ofstream::out);
    matlabFile.precision(4);
    matlabFile << "close all;\nfigure ();\nhold on;\ngrid on;";
    matlabFile << "\nset(gca,'LineStyleOrder', '--');";
    matlabFile << "\ntext(1,0.685 ,'Threshold' , 'FontSize',15,'FontWeight','bold','Interpreter','Latex');";

    matlabFile << "\nlocalF1= [";
    
    std::string value;
    std::ifstream Readlocal_F1 (local_F1_vs_learned_category.c_str());
    std::getline (Readlocal_F1, value);
    matlabFile << value;

    int iteration = 1;
    while (Readlocal_F1.good())
    {
		std::getline (Readlocal_F1, value);
		if(value.empty ()) // Skip blank lines or comments
			continue;
			
		matlabFile <<", " << value ;
		iteration++;
    }
    
    int remain = iteration % 5;
    matlabFile << "];\nline([0 "<< iteration + 5 - remain <<"],["<< P_Threshold << " " <<P_Threshold <<"] ,'Color',[0 0 0], 'LineWidth',1);";
    matlabFile << "\naxis([0,"<< iteration + 5 - remain <<",0.5,1.05]);"; 
    matlabFile << "\nplot(1:size(localF1,2), localF1(1:size(localF1,2)),'-.O', 'Color',[1 0 1], 'LineWidth',2.);";
    matlabFile << "\nxlabel('Number of Learned Categories','FontSize',15);";
    matlabFile << "\nylabel('Protocol Accuracy','FontSize',15);";

    matlabFile.close();
    ROS_INFO("\t\tMATLAB file created...");
}


void plotGlobalF1VsNumberOfLearnedCategoriesInMatlab(int RunCount, 
								    				  string global_F1_vs_learned_category)
{
    // If the text inside the plot is not well-appear, please install the font in ubuntu using the following commands.
    // sudo apt-get install xfonts-75dpi
    // sudo apt-get install xfonts-75dpi
    
    char run_count [10];
    sprintf( run_count, "%d",RunCount );
    string path = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/global_F1_vs_learned_category.m";
        
    ROS_INFO("\t\tpath = %s", path.c_str());

    ofstream matlabFile;
    matlabFile.open (path.c_str(), std::ofstream::out);
    matlabFile.precision(4);
    matlabFile << "close all;\nfigure ();\nhold on;\ngrid on;";
    matlabFile << "\nset(gca,'LineStyleOrder', '--');";
    // matlabFile << "\ntext(1,0.685 ,'Threshold' , 'FontSize',15,'FontWeight','bold','Interpreter','Latex');";

    matlabFile << "\nglobalF1= [";
    
    std::string value;
    std::ifstream ReadGlobalF1 (global_F1_vs_learned_category.c_str());
    std::getline (ReadGlobalF1, value);
    matlabFile << value;

    int iteration = 1;
    while (ReadGlobalF1.good())
    {
		std::getline (ReadGlobalF1, value);
		if(value.empty ()) // Skip blank lines or comments
			continue;
		//  ROS_INFO("value = %s", value.c_str());
		matlabFile <<", " << value ;
		iteration++;
    }
    
    int remain = iteration % 5;
    matlabFile << "];\naxis([0,"<< iteration + 5 - remain <<",0.5,1.05]);"; 
    matlabFile << "\nplot(1:size(globalF1,2), globalF1(1:size(globalF1,2)),'-.O', 'Color',[0 0 1], 'LineWidth',2.);";
    matlabFile << "\nxlabel('Number of Learned Categories','FontSize',15);";
    matlabFile << "\nylabel('Global Classification Accuracy','FontSize',15);";
    matlabFile.close();
    ROS_INFO("\t\tMATLAB file created...");
}



void plotNumberOfLearnedCategoriesVsIterationsInMatlab( int RunCount, 
														string Number_of_learned_categories_vs_Iterations)
{
 
    // If the text inside the plot is not well-appear, please install the font in ubuntu using the following commands.
    // sudo apt-get install xfonts-75dpi
    // sudo apt-get install xfonts-75dpi
    
    char run_count [10];
    sprintf( run_count, "%d",RunCount );
    string path = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/number_of_learned_categories_vs_Iterations.m";
        
    ROS_INFO("\t\tpath = %s", path.c_str());

    ofstream matlabFile;
    matlabFile.open (path.c_str(), std::ofstream::out);
    matlabFile.precision(4);
    matlabFile << "figure ();\nhold on;\ngrid on;";
    matlabFile << "\nset(gca,'LineStyleOrder', '-');";
    matlabFile << "\nNLI= [";
    
    int iteration = 1;
    std::string value;
    std::ifstream ReadNLI (Number_of_learned_categories_vs_Iterations.c_str());
    std::getline (ReadNLI, value);
    matlabFile << iteration;

//     int iteration = 2;
    while (ReadNLI.good())
    {
	std::getline (ReadNLI, value);
	if(value.empty ()) // Skip blank lines or comments
	    continue;	
	if (strcmp(value.c_str(),"1")==0)
	{
	    matlabFile << "," << iteration ;	    
	    iteration++;
	}
	else
	{
	    iteration++;
	}
    }
    matlabFile << "];";
    matlabFile << "\nplot(NLI(1:size(NLI,2)), 1:size(NLI,2), '--O', 'Color',[1 0 0], 'LineWidth',2.)";
    matlabFile << "\nxlabel('Question / Correction Iterations','FontSize',15);";
    matlabFile << "\nylabel('Number of Learned Categories','FontSize',15);";
    matlabFile.close();
    ROS_INFO("\t\tMATLAB file created...");
}

void plotNumberOfStoredInstancesPerCategoryInMatlab( vector <ObjectCategory> list_of_object_category)
{
 
	string path = ros::package::getPath("rug_simulated_user")+ "/result/experiment_1/number_of_stored_instaces_per_category.m";
    ofstream matlabFile;
    matlabFile.open (path.c_str(), std::ofstream::out);
    matlabFile.precision(4);
    matlabFile << "figure ();\nhold on;\ngrid on;";
    matlabFile << "\nNIC= [";
	
	string path_tmp = home_directory_address+ "/Category/Category.txt";
    std::ifstream category_name (path_tmp.c_str());
    string cat_name ="";

	int counter = 0;
	while ((category_name.good()) && 
			(counter < list_of_object_category.size()))
    {
		int idx = 0;
		std::getline (category_name, cat_name);
 	   	cat_name = extractCategoryName(cat_name);
		while ((list_of_object_category.at(idx).cat_name.c_str() != cat_name) && 
		 		(idx < list_of_object_category.size()))
		{
			idx ++;
		}

		if (counter < list_of_object_category.size()-1)
		{
			matlabFile << list_of_object_category.at(idx).rtov_keys.size()<<",";
		}
		else
		{
			matlabFile << list_of_object_category.at(idx).rtov_keys.size()<<"];\n";
		}		
		counter ++;
	}
	
	matlabFile << "list = {'";
	std::ifstream category_name_tmp (path_tmp.c_str());
	for (size_t i = 0; i < list_of_object_category.size(); i++) // retrieves all categories from perceptual memory
	{
		int idx = 0;
		std::getline (category_name_tmp, cat_name);
 	   	cat_name = extractCategoryName(cat_name);
		int lfind =  cat_name.find("_");
    	if (lfind > 0) 
			cat_name.replace(lfind, 1,1, '-');

		if (i < list_of_object_category.size()-1)
		{
			matlabFile << cat_name<<"', '";
		}
		else
		{
			matlabFile << cat_name<<"'};\n";
		}			
	}
	matlabFile << "categories = categorical(list,list);\n";
	matlabFile << "bar (categories, NIC);\n";
    matlabFile << "ylabel('Number of Stored Instances','FontSize',15);";
    matlabFile.close();
    ROS_INFO("\t\tMATLAB file created...");

    ROS_INFO("\t\t cleaning log files...");

	string system_command= "rm " + ros::package::getPath("rug_simulated_user") + "/result/experiment_1/Category.txt";
	system( system_command.c_str());

	system_command= "rm " + ros::package::getPath("rug_simulated_user") + "/result/experiment_1/Category_Introduced.txt";
	system( system_command.c_str());
	
	system_command= "rm " + ros::package::getPath("rug_simulated_user") + "/result/experiment_1/f1_vs_learned_category.txt";
	system( system_command.c_str());

	system_command= "rm " + ros::package::getPath("rug_simulated_user") + "/result/experiment_1/local_f1_vs_learned_category.txt";
	system( system_command.c_str());

	system_command= "rm " + ros::package::getPath("rug_simulated_user") + "/result/experiment_1/PrecisionMonitor.txt";
	system( system_command.c_str());

}




int IntroduceNewInstanceVFH ( std::string PCDFileAddress,
			    unsigned int cat_id, 
			    unsigned int track_id,
			    unsigned int view_id
			    )
{

//     string categoryName = PCDFileAddress;
//     categoryName.resize(13);
    
    string categoryName = extractCategoryName(PCDFileAddress);

    
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#' || categoryName == "Category//Unk") // Skip blank lines or comments
    {
	return 0;
    }
    
    PCDFileAddress = home_directory_address +"/"+ PCDFileAddress.c_str();

    //load a PCD object  
    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
    if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *PCDFile) == -1)
    {	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
    }
    else
    {
	    ROS_INFO("\t\t[1]-Loaded a point cloud: %s", PCDFileAddress.c_str());
    }
    
    //compute Spin Image for given point clould
    //Declare a boost share ptr to the SITOV msg
    boost::shared_ptr< vector <SITOV> > objectViewVFHs;
    objectViewVFHs = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
    
    //Call the library function for estimateVFH
    estimateVFH( PCDFile, 
		0.01 /*downsampling_voxel_size*/, 
		0.05 /*normal_estimation_radius*/,
		objectViewVFHs,
		0
		);

    double Cat_ICD=0.0001;
    putObjectViewSpinImagesInSpecificCategory(categoryName,cat_id,track_id,view_id,objectViewVFHs,Cat_ICD);

    ROS_INFO("\t\t[-]-%s created...",categoryName.c_str());
        
    return (0);
}

int introduceNewCategoryVFH(int class_index,
			    unsigned int &track_id,
			    unsigned int &instance_number,
			    string fname
			   )
{
	string instance_path;
	for(int i = 0; i < 3 ; i++)  // 2 instances would be enough
	{
// 	    selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path);
	    if (selectAnInstancefromSpecificCategory(class_index, instance_number, instance_path)==-1)
	    {	
		ROS_ERROR("\t\t[-]-The object view or category does not exist");
		return -1;// ERROR : file doesn't exist
	    }
	    ROS_INFO("\t\t[-]-Instance number %ld", instance_number);
	    int cat_id = 1; // cat_id is always 1 beacuse we use cat_name key:<cat_name><cat_id>
	    int view_id = 1; // view_id is always 1 beacuse we use TID key:<TID><VID>
	    IntroduceNewInstanceVFH(instance_path, cat_id, track_id, view_id); 
	    track_id++;
	    //view_id ++; // in this implementation we consider VID as a constant
	}
	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 
 	report_category_introduced(fname,categoryName.c_str());
	return 0;
}


bool fexists(std::string filename) 
{
  ifstream ifile(filename.c_str());
  return ifile.good();
}

int sum_all_experiments_results ( int iterations,
				  float Success_Precision		,			    
				  float average_class_precision_value,
				  float number_of_stored_instances, 
				  int number_of_taught_categories,
				  string name_of_approach
				  )
{
    vector <float> tmp;
    tmp.push_back(float (iterations)); 
    tmp.push_back(float (number_of_taught_categories)); 
    tmp.push_back(float (number_of_stored_instances)); 
    tmp.push_back(Success_Precision); 
    tmp.push_back(average_class_precision_value); 

    std::string sumResultsOfExperiments;
    sumResultsOfExperiments = ros::package::getPath("rug_simulated_user")+ "/result/sum_all_results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",sumResultsOfExperiments.c_str() );
    int exp_num =1;
    if (!fexists(sumResultsOfExperiments.c_str()))
    {
		ROS_INFO("File not exist");	
		std::ofstream sum_results_of_experiments;
		sum_results_of_experiments.open (sumResultsOfExperiments.c_str(), std::ofstream::out);
		sum_results_of_experiments.precision(4);
		sum_results_of_experiments <<iterations <<"\n"<< number_of_taught_categories <<"\n"<< number_of_stored_instances/(float) number_of_taught_categories <<"\n"<< Success_Precision<< "\n"<< average_class_precision_value;
		sum_results_of_experiments.close();
    }
	else
    {
		ROS_INFO("File exist");
		string tmp_value;
		std::ifstream results_of_experiments;
		results_of_experiments.open (sumResultsOfExperiments.c_str());
		vector <float> value;
		for (int i=0; (results_of_experiments.good()) && (i < tmp.size()) ; i++)
		{
			std::getline (results_of_experiments, tmp_value);
			if(tmp_value.empty () || tmp_value.at (0) == '#') // Skip blank lines or comments
			continue;
			value.push_back(atof(tmp_value.c_str())+tmp.at(i));
		    // ROS_INFO("tmp_value.c_str() = %s",tmp_value.c_str() );
		    // ROS_INFO("atof -> tmp_value.c_str() = %f",atof(tmp_value.c_str()) );
		    // ROS_INFO("tmp[%i] = %f",i, tmp.at(i) );
		    // ROS_INFO("value[%i]= %f",i, value.at(i) );			
		}
		results_of_experiments.close();

		// 	ROS_INFO("number_of_exist_experiments = %i",number_of_exist_experiments );

		std::ofstream sum_results_of_experiments;
		sum_results_of_experiments.open (sumResultsOfExperiments.c_str(), std::ofstream::out);
		sum_results_of_experiments.precision(4);
		sum_results_of_experiments <<value.at(0) <<"\n"<< value.at(1) <<"\n"<< value.at(2) <<"\n"<< value.at(3)<< "\n"<< value.at(4);
		sum_results_of_experiments.close();
    } 

return 0;	
    
}

int average_all_experiments_results ( int total_number_of_experiments,
				      string name_of_approach)
{
    std::string sumResultsOfExperiments;
    sumResultsOfExperiments = ros::package::getPath("rug_simulated_user")+ "/result/sum_all_results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",sumResultsOfExperiments.c_str() );
    
    if (!fexists(sumResultsOfExperiments.c_str()))
    {
	ROS_INFO("Expriments summary file of not exist");	
    }else
    {
	ROS_INFO("File exist");
	string tmp_value;
	std::ifstream read_results_of_experiments;
	read_results_of_experiments.open (sumResultsOfExperiments.c_str());
	vector <float> value;
	for (int i=0; read_results_of_experiments.good(); i++)
	{
	    std::getline (read_results_of_experiments, tmp_value);
	    if(tmp_value.empty () || tmp_value.at (0) == '#') // Skip blank lines or comments
		continue;
	    value.push_back(atof(tmp_value.c_str())/total_number_of_experiments);	    
	}
	read_results_of_experiments.close();

	std::string resultsOfExperiments;
	resultsOfExperiments = ros::package::getPath("rug_simulated_user")+ "/result/results_of_"+name_of_approach+"_experiments.txt";
	std::ofstream results_of_experiments;
	results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::app);
	results_of_experiments.precision(4);
	results_of_experiments << "\n"<<"Avg."<<"\t"<<value.at(0) <<"\t\t"<< value.at(1) <<"\t\t"<< value.at(2) <<"\t\t"<< value.at(3)<< "\t\t"<< value.at(4);
	results_of_experiments << "\n---------------------------------------------------------------------------";
	results_of_experiments.close();
    }
    return 0;
}

int reportAllExperimentalResults (int TP, int FP, int FN,int obj_num,			    
				    vector <float> average_class_precision,
				    float number_of_stored_instances, 
				    int number_of_taught_categories,
				    string name_of_approach
				   )
{
    //ROS_INFO("TEST report_all_experiments_results fucntion");
    int total_number_of_experiments = 50;//TODO: shold be input parameres
    unsigned int number_of_exist_experiments = 0;
    
    float average_class_precision_value = 0;
    for (int i = 0; i < average_class_precision.size(); i++)
    {
		average_class_precision_value += average_class_precision.at(i); 	    
    }
    average_class_precision_value = average_class_precision_value / average_class_precision.size();
	
    std::string resultsOfExperiments;
    resultsOfExperiments = ros::package::getPath("rug_simulated_user")+ "/result/results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",resultsOfExperiments.c_str() );
    int exp_num =1;
    double Success_Precision = 0;
    if (!fexists(resultsOfExperiments.c_str()))
    {
	ROS_INFO("File not exist");
	Success_Precision = TP/double (TP+FP);
	double Recall = TP/double (TP+FN);

	std::ofstream results_of_experiments;
	results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::out);
	results_of_experiments.precision(4);
	results_of_experiments << "\nNum"<<"\tIterations" <<"\t"<< "Categories" <<"\t"<< "Instances"<< "\t"<< "GS" << "\t\t"<< "ACS";
	results_of_experiments << "\n---------------------------------------------------------------------------";
	results_of_experiments << "\n"<<exp_num<<"\t"<<obj_num <<"\t\t"<< number_of_taught_categories <<"\t\t"<< number_of_stored_instances/(float)number_of_taught_categories <<"\t\t"<< Success_Precision<< "\t\t"<< average_class_precision_value;
	results_of_experiments << "\n---------------------------------------------------------------------------";
	results_of_experiments.close();
	results_of_experiments.clear();
    }else
    {
	ROS_INFO("File exist");
	string tmp;
	std::ifstream num_results_of_experiments;
	num_results_of_experiments.open (resultsOfExperiments.c_str());
	while (num_results_of_experiments.good()) 
	{
	    std::getline (num_results_of_experiments, tmp);
	    if(tmp.empty () || tmp.at (0) == '#') // Skip blank lines or comments
		continue;
	    number_of_exist_experiments++;
	}
	num_results_of_experiments.close();
	ROS_INFO("number_of_exist_experiments = %i",number_of_exist_experiments );
	exp_num = (number_of_exist_experiments)/2;
	ROS_INFO("(number_of_exist_experiments)/2 = %i",exp_num );

	Success_Precision = TP/double (TP+FP);
	double Recall = TP/double (TP+FN);
	std::ofstream results_of_experiments;
	results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::app);
	results_of_experiments.precision(4);
	results_of_experiments << "\n"<<exp_num<<"\t"<<obj_num <<"\t\t"<< number_of_taught_categories <<"\t\t"<< number_of_stored_instances/(float)number_of_taught_categories <<"\t\t"<< Success_Precision<< "\t\t"<< average_class_precision_value;
	results_of_experiments << "\n---------------------------------------------------------------------------";
	results_of_experiments.close();
    } 
    
    sum_all_experiments_results ( obj_num,
				  Success_Precision,			    
				  average_class_precision_value,
				  number_of_stored_instances, 
				  number_of_taught_categories,
				  name_of_approach );

    if (exp_num == total_number_of_experiments)
    {
		average_all_experiments_results (total_number_of_experiments, name_of_approach);
    }
    
return 0;	
}


int report_all_context_change_experiments_results (int TP, int FP, int FN,int itr_num,
						    int context_change_idx,int context_change_itr,
						    vector <float> average_class_precision,
						    float number_of_stored_instances, 
						    int number_of_taught_categories,
						    string name_of_approach,
						    int total_number_of_experiments
						  )
{
    //ROS_INFO("TEST report_all_experiments_results fucntion");
    unsigned int number_of_exist_experiments = 0;
    
    float average_class_precision_value =0;
    for (int i =0; i<average_class_precision.size(); i++)
    {
	average_class_precision_value+=average_class_precision.at(i); 	    
    }
    average_class_precision_value=average_class_precision_value/average_class_precision.size();
	
    std::string resultsOfExperiments;
    resultsOfExperiments = ros::package::getPath("rug_simulated_user")+ "/result/results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",resultsOfExperiments.c_str() );
    int exp_num =1;
    double Success_Precision = 0;
    if (!fexists(resultsOfExperiments.c_str()))
    {
	ROS_INFO("File not exist");
	Success_Precision = TP/double (TP+FP);
	double Recall = TP/double (TP+FN);

	std::ofstream results_of_experiments;
	results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::out);
	results_of_experiments.precision(4);
	results_of_experiments << "\nNum"<<"\tIterations" <<"\t" << "context_change(Idx,Itr)" << "\t"<< "Categories" <<"\t"<< "Instances"<< "\t"<< "GS" << "\t\t"<< "ACS";
	results_of_experiments << "\n---------------------------------------------------------------------------------------------------------";
	results_of_experiments << "\n"<<exp_num<<"\t"<<itr_num <<"\t\t("<<context_change_idx<<","<<context_change_itr<<")\t\t\t"<< number_of_taught_categories <<"\t\t"<< number_of_stored_instances/(float)number_of_taught_categories <<"\t\t"<< Success_Precision<< "\t\t"<< average_class_precision_value;
	results_of_experiments << "\n---------------------------------------------------------------------------------------------------------";
	results_of_experiments.close();
	results_of_experiments.clear();
    }else
    {
	ROS_INFO("File exist");
	string tmp;
	std::ifstream num_results_of_experiments;
	num_results_of_experiments.open (resultsOfExperiments.c_str());
	while (num_results_of_experiments.good()) 
	{
	    std::getline (num_results_of_experiments, tmp);
	    if(tmp.empty () || tmp.at (0) == '#') // Skip blank lines or comments
		continue;
	    number_of_exist_experiments++;
	}
	num_results_of_experiments.close();
	ROS_INFO("number_of_exist_experiments = %i",number_of_exist_experiments );
	exp_num = (number_of_exist_experiments)/2;
	ROS_INFO("(number_of_exist_experiments)/2 = %i",exp_num );

	Success_Precision = TP/double (TP+FP);
	double Recall = TP/double (TP+FN);
	std::ofstream results_of_experiments;
	results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::app);
	results_of_experiments.precision(4);
	results_of_experiments << "\n"<<exp_num<<"\t"<<itr_num <<"\t\t("<<context_change_idx<<","<<context_change_itr<<")\t\t"<< number_of_taught_categories <<"\t\t"<< number_of_stored_instances/(float)number_of_taught_categories <<"\t\t"<< Success_Precision<< "\t\t"<< average_class_precision_value;
	results_of_experiments << "\n---------------------------------------------------------------------------------------------------------";
	results_of_experiments.close();
    } 
    
    sum_all_experiments_results ( itr_num,
				  Success_Precision,			    
				  average_class_precision_value,
				  number_of_stored_instances, 
				  number_of_taught_categories,
				  name_of_approach );
    if (exp_num == total_number_of_experiments)
    {
	average_all_experiments_results (total_number_of_experiments,
					 name_of_approach);
    }
    
return 0;	
}


int read_a_number_from_file ( string pakage_name,
			       string file_name)
{
    std::string path;
    path = ros::package::getPath(pakage_name.c_str())+ "/result/"+file_name;
    ROS_INFO("file path = %s",path.c_str() );
    string tmp_value;
    std::ifstream fnumber;
    int number =0;
    fnumber.open (path.c_str());
    for (int i=0; (fnumber.good()); i++)
    {
	std::getline (fnumber, tmp_value);
	if(tmp_value.empty () || tmp_value.at (0) == '#') // Skip blank lines or comments
	    continue;
	number =  atoi(tmp_value.c_str());
    }
    fnumber.close();

return number;	
    
}

int write_a_number_to_file ( string pakage_name,
			       string file_name,
			       int number )
{
    std::string path;
    path = ros::package::getPath(pakage_name.c_str())+ "/result/"+file_name;
    ROS_INFO("file path = %s",path.c_str() );
    std::ofstream fnumber;
    fnumber.open (path.c_str(), std::ofstream::trunc);
    fnumber.precision(4);
    fnumber << number;
    fnumber.close();

return 0;	
    
}

//TODO : should be moved to functionality.cpp
////////////////////////////////////////////////////////////////////////////////////////////////////
int chiSquaredDistanceBetweenTwoObjectViewHistogram (SITOV objectViewHistogram1,
						    SITOV objectViewHistogram2, 
						    float &diffrence)
{
	if (objectViewHistogram1.spin_image.size() ==  objectViewHistogram2.spin_image.size())
	{
		diffrence = 0;
		for (size_t i = 0; i < objectViewHistogram1.spin_image.size(); i++)
		{
		  if (objectViewHistogram1.spin_image.at(i) + objectViewHistogram2.spin_image.at(i) > 0)
		  {
			  diffrence += pow( (objectViewHistogram1.spin_image.at(i) - objectViewHistogram2.spin_image.at(i)) , 2) / 
				      (objectViewHistogram1.spin_image.at(i) + objectViewHistogram2.spin_image.at(i)) ;
		  }
		}
		diffrence = 0.5 * diffrence;
		return(1);
	}
	else 
	{
		ROS_INFO("\t\t[-]- object1 size = %ld", objectViewHistogram1.spin_image.size());
		ROS_INFO("\t\t[-]- object2 size = %ld", objectViewHistogram2.spin_image.size());
		ROS_ERROR("Can not compare two object view histograms with diffrent lenght");
		return(0);
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int chiSquaredBasedObjectCategoryDistance( SITOV target,
		vector< SITOV > category_instances,
		float &minimumDistance, 
		int &best_matched_index, 
		PrettyPrint &pp)

{
	size_t category_size = category_instances.size();  
	best_matched_index=-1;//not matched

	if (category_size < 1)
	{
		pp.warn(std::ostringstream().flush() <<  "Error: Size of category is zero - could not compute objectCategoryDistance D(t,C)");
		// return 0;
	}
	else
	{
		// find the minimum distance between target object and category instances 
		std::vector<float> listOfDiffrence;
		float minimum_distance =10000000;
		for (size_t i=0; i<category_size; i++)
		{
			float tmp_diff =0;
			SITOV categoryInstance;
			categoryInstance = category_instances.at(i);

			chiSquaredDistanceBetweenTwoObjectViewHistogram(target,categoryInstance, tmp_diff);
			listOfDiffrence.push_back(tmp_diff);
			if (tmp_diff < minimum_distance)
			{
				minimum_distance=tmp_diff;
				best_matched_index=i;
			}    
		}

		pp.info(std::ostringstream().flush() << "D(target,category) ="<< minimum_distance);

		minimumDistance=minimum_distance;
		// 		pp.printCallback();
	}

	return 1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////// adde K&A
int FidelityDistanceBetweenTwoObjectViewHistogram (SITOV objectViewHistogram1,
						    SITOV objectViewHistogram2, 
						    float &diffrence)
{
	if (objectViewHistogram1.spin_image.size() ==  objectViewHistogram2.spin_image.size())
	{
		diffrence = 0;
		for (size_t i = 0; i < objectViewHistogram1.spin_image.size(); i++)
		{
		  if (objectViewHistogram1.spin_image.at(i) + objectViewHistogram2.spin_image.at(i) > 0)
		  {
			  diffrence -= sqrt((objectViewHistogram1.spin_image.at(i) * objectViewHistogram2.spin_image.at(i)));  
				      
		  }
		}
		//diffrence = 0.5 * diffrence;
		return(1);
	}
	else 
	{
		ROS_INFO("\t\t[-]- object1 size = %ld", objectViewHistogram1.spin_image.size());
		ROS_INFO("\t\t[-]- object2 size = %ld", objectViewHistogram2.spin_image.size());
		ROS_ERROR("Can not compare two object view histograms with diffrent lenght");
		return(0);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////// adde K&A
int FidelityBasedObjectCategoryDistance( SITOV target,
		vector< SITOV > category_instances,
		float &minimumDistance, 
		int &best_matched_index, 
		PrettyPrint &pp)

{
	size_t category_size = category_instances.size();  
	best_matched_index=-1;//not matched

	if (category_size < 1)
	{
		pp.warn(std::ostringstream().flush() <<  "Error: Size of category is zero - could not compute objectCategoryDistance D(t,C)");
		// return 0;
	}
	else
	{
		// find the minimum distance between target object and category instances 
		std::vector<float> listOfDiffrence;
		float minimum_distance =10000000;
		for (size_t i=0; i<category_size; i++)
		{
			float tmp_diff =0;
			SITOV categoryInstance;
			categoryInstance = category_instances.at(i);

			FidelityDistanceBetweenTwoObjectViewHistogram(target,categoryInstance, tmp_diff);
			listOfDiffrence.push_back(tmp_diff);
			if (tmp_diff < minimum_distance)
			{
				minimum_distance=tmp_diff;
				best_matched_index=i;
			}    
		}
		minimumDistance=minimum_distance;
	}

	return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int squaredChordDistanceBetweenTwoObjectViewHistogram (SITOV objectViewHistogram1,
						    SITOV objectViewHistogram2, 
						    float &diffrence)
{
	if (objectViewHistogram1.spin_image.size() ==  objectViewHistogram2.spin_image.size())
	{
		diffrence = 0;
		for (size_t i = 0; i < objectViewHistogram1.spin_image.size(); i++)
		{
		  if (objectViewHistogram1.spin_image.at(i) + objectViewHistogram2.spin_image.at(i) > 0)
		  {
			  diffrence += pow((sqrt(objectViewHistogram1.spin_image.at(i)) - sqrt(objectViewHistogram2.spin_image.at(i))) , 2);
		  }
		}
		diffrence = 0.5 * diffrence;
		return(1);
	}
	else 
	{
		ROS_INFO("\t\t[-]- object1 size = %ld", objectViewHistogram1.spin_image.size());
		ROS_INFO("\t\t[-]- object2 size = %ld", objectViewHistogram2.spin_image.size());
		ROS_ERROR("Can not compare two object view histograms with diffrent lenght");
		return(0);
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int squaredChordBasedObjectCategoryDistance( SITOV target,
		vector< SITOV > category_instances,
		float &minimumDistance, 
		int &best_matched_index, 
		PrettyPrint &pp)

{
	size_t category_size = category_instances.size();  
	best_matched_index=-1;//not matched

	if (category_size < 1)
	{
		pp.warn(std::ostringstream().flush() <<  "Error: Size of category is zero - could not compute objectCategoryDistance D(t,C)");
		// return 0;
	}
	else
	{
		// find the minimum distance between target object and category instances 
		std::vector<float> listOfDiffrence;
		float minimum_distance =10000000;
		for (size_t i=0; i<category_size; i++)
		{
			float tmp_diff =0;
			SITOV categoryInstance;
			categoryInstance = category_instances.at(i);

			squaredChordDistanceBetweenTwoObjectViewHistogram(target,categoryInstance, tmp_diff);
			listOfDiffrence.push_back(tmp_diff);
			if (tmp_diff < minimum_distance)
			{
				minimum_distance=tmp_diff;
				best_matched_index=i;
			}    
		}
		minimumDistance=minimum_distance;
	}

	return 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int diffrenceBetweenTwoObjectViewHistogram(SITOV objectViewHistogram1,
		SITOV objectViewHistogram2, 
		float &diffrence)
{
	if (objectViewHistogram1.spin_image.size() ==  objectViewHistogram2.spin_image.size())
	{
		diffrence =0;
		for (size_t i = 0; i < objectViewHistogram1.spin_image.size(); i++)
		{
			diffrence += pow( (objectViewHistogram1.spin_image.at(i) - objectViewHistogram2.spin_image.at(i)) , 2);
			// 	    diffrence += log (pow( (objectViewHistogram1.spin_image.at(i) - objectViewHistogram2.spin_image.at(i)) , 2));
		}
		// 	diffrence = log (diffrence);
		return(1);
	}
	else 
	{
		ROS_INFO("\t\t[-]- object1 size = %ld", objectViewHistogram1.spin_image.size());
		ROS_INFO("\t\t[-]- object2 size = %ld", objectViewHistogram2.spin_image.size());
		ROS_ERROR("Can not compare two object view histograms with diffrent lenght");
		return(0);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int euclideanBasedObjectCategoryDistance( SITOV target,
		vector< SITOV > category_instances,
		float &minimumDistance, 
		int &best_matched_index, 
		PrettyPrint &pp)

{
	size_t category_size = category_instances.size();  
	best_matched_index=-1;//not matched

	if (category_size < 1)
	{
		pp.warn(std::ostringstream().flush() <<  "Error: Size of category is zero - could not compute objectCategoryDistance D(t,C)");
		// return 0;
	}
	else
	{
		// find the minimum distance between target object and category instances 
		std::vector<float> listOfDiffrence;
		float minimum_distance =10000000;
		for (size_t i=0; i<category_size; i++)
		{
			float tmp_diff =0;
			SITOV categoryInstance;
			categoryInstance = category_instances.at(i);

			diffrenceBetweenTwoObjectViewHistogram(target,categoryInstance, tmp_diff);
			//pp.info(std::ostringstream().flush() <<"diffrenceBetweenTwoObjectViewHistogram [target, Instance "<< i<<"]= "<< tmp_diff);
			listOfDiffrence.push_back(tmp_diff);
			if (tmp_diff < minimum_distance)
			{
				minimum_distance=tmp_diff;
				best_matched_index=i;
			}    
		}
		//pp.info(std::ostringstream().flush() << "D(target,category) ="<< minimum_distance);

		minimumDistance=minimum_distance;
	}

	return 1;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
int kLdiffrenceBetweenTwoObjectViewHistogram(SITOV objectViewHistogram1,
					      SITOV objectViewHistogram2, 
					      double &diffrence)
{
	
	//https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
	if (objectViewHistogram1.spin_image.size() ==  objectViewHistogram2.spin_image.size())
	{
		diffrence =10000000;
		double distance_P_Q =0;
		double distance_Q_P =0;
		
		for (size_t i = 0; i < objectViewHistogram1.spin_image.size(); i++)
		{
			if ((objectViewHistogram1.spin_image.at(i) == 0) || (objectViewHistogram2.spin_image.at(i) == 0) )
			{
				continue;
				//objectViewHistogram1.spin_image.at(i) = 0.00001; // maybe we should skip this element
			}
			distance_P_Q += objectViewHistogram1.spin_image.at(i) * log10 (objectViewHistogram2.spin_image.at(i)/objectViewHistogram1.spin_image.at(i));
			distance_P_Q += objectViewHistogram2.spin_image.at(i) * log10 (objectViewHistogram1.spin_image.at(i)/objectViewHistogram2.spin_image.at(i));
		}
		diffrence = -0.5 * (distance_P_Q + distance_Q_P);
		return(1);
	}
	else 
	{
		return(0);
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int kLBasedObjectCategoryDistance( SITOV target,
		vector< SITOV > category_instances,
		float &minimumDistance, 
		int &best_matched_index, 
		PrettyPrint &pp)

{
	size_t category_size = category_instances.size();  
	best_matched_index=-1;//not matched

	if (category_size < 1)
	{
		pp.warn(std::ostringstream().flush() <<  "Error: Size of category is zero - could not compute objectCategoryDistance D(t,C)");
		// return 0;
	}
	else
	{
		// find the minimum distance between target object and category instances 
		std::vector<float> listOfDiffrence;
		float minimum_distance =10000000;
		for (size_t i=0; i<category_size; i++)
		{
			double tmp_diff = 0;
			SITOV categoryInstance;
			categoryInstance = category_instances.at(i);

			kLdiffrenceBetweenTwoObjectViewHistogram(target,categoryInstance, tmp_diff);
			
			listOfDiffrence.push_back(tmp_diff);
			if (tmp_diff < minimum_distance)
			{
				minimum_distance=tmp_diff;
				best_matched_index=i;
			}    
		}

		//ROS_INFO("\t\t[-]-objectCategoryDistance D(target,category) is: %f ", minimum_distance);
		//pp.info(std::ostringstream().flush() << "D(target,category) ="<< minimum_distance);

		minimumDistance=minimum_distance;
		//pp.printCallback();
	}

	return 1;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////




