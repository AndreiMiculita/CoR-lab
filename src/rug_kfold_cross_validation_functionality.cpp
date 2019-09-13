#ifndef _KFOLD_CROSS_VALIDATION_LIB_CPP_
#define _KFOLD_CROSS_VALIDATION_LIB_CPP_

#define recognitionThershold 2


/* _________________________________
   |                                 |
   |           INCLUDES              |
   |_________________________________| */

//system includes
#include <std_msgs/String.h>
#include <sstream>
#include <vector>
#include <iostream>
#include <fstream>

//ros includes 
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>


//perception db includes
#include <race_perception_msgs/perception_msgs.h>
#include <race_perception_db/perception_db.h>
#include <race_perception_db/perception_db_serializer.h>

// Need to include the pcl ros utilities
//package includes
#include <object_descriptor/object_descriptor_functionality.h>
#include <race_3d_object_tracking/TrackedObjectPointCloud.h>
#include <object_conceptualizer/object_conceptualization.h>
#include <feature_extraction/spin_image.h>
#include <race_perception_utils/cycle.h>
#include <race_perception_utils/print.h>


//pcl includes
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ros/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/esf.h>
#include <pcl/features/gfpfh.h>
#include <pcl/features/vfh.h>

//GRSD needs PCL 1.8.0
#include <pcl/features/grsd.h>





#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <sys/time.h>  
#include <iostream>
#include <dirent.h>  //I use dirent.h for get folders in directory which is also available for windows:

#include <race_deep_learning_feature_extraction/deep_representation.h>


/* _________________________________
  |                                 |
  |         Global variable         |
  |_________________________________| */

  
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointXYZRGBA T;  
PerceptionDB* _pdb; //initialize the class

int track_id_gloabal =1;
using namespace pcl;
using namespace std;
using namespace ros;

// std::string home_address= "/home/hamidreza/";

 string extractObjectName (string object_name_orginal )
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

string extractCategoryName (string instance_path )
{
    //ROS_INFO("\t\t instance_path = %s", InstancePath.c_str());
    string category_name="";	    
    int ffind = instance_path.find("//")+2;  
    int lfind =  instance_path.rfind("_Cat");       
    for (int i=0; i<(lfind-ffind); i++)
    {
		category_name += instance_path.at(i+ffind);
    }
    return (category_name);
}


int estimateESFDescription (boost::shared_ptr<PointCloud<PointT> > cloud, 
			     			pcl::PointCloud<pcl::ESFSignature640>::Ptr &descriptor)
{
    // ESF estimation object.
    pcl::ESFEstimation<PointT, pcl::ESFSignature640> esf;
    esf.setInputCloud(cloud);
    esf.compute(*descriptor);

    return 0;  
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int addObjectViewHistogramInSpecificCategory(	std::string cat_name, unsigned int cat_id, 
												unsigned int track_id, unsigned int view_id, 
												SITOV objectViewHistogram , PrettyPrint &pp
												)
{
	//// add representation of an object to perceptual memory
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

	//// put slice to the db
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
    //ROS_INFO("\t\t[-]v_key: %s, view_id: %i, track_id: %i", v_key.c_str(), view_id, track_id);

    //// Put one view to the db
    _pdb->put(v_key, v_s);
	//ROS_INFO("****========***** add an object view to PDB took = %f", (ros::Time::now() - start_time).toSec());

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
    _oc.icd = 1.0f;
    oc_size = ros::serialization::serializationLength(_oc);

    //pp.info(std::ostringstream().flush() << _oc.cat_name.c_str() << " category has " << _oc.rtov_keys.size() << " objects.");

    boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
    PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
    leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
    _pdb->put(oc_key, ocs);
	//ROS_INFO("****========***** update a category took = %f", (ros::Time::now() - start_time).toSec());
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
    pp.info(std::ostringstream().flush() << "ICD for " << _oc.cat_name.c_str() << " category updated. New ICD is: "<< _oc.icd);

    boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
    PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
    leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
    _pdb->put(oc_key, ocs);
	//ROS_INFO("****========***** update a CATEGORY to PDB took = %f", (ros::Time::now() - start_time).toSec());
	//ROS_INFO("****========*********========*********========*********========*********========*****");
    return (1);
}

int conceptualizeObjectViewSpinImagesInSpecificCategory(std::string cat_name, unsigned int cat_id, 
														unsigned int track_id, unsigned int view_id, 
														vector <SITOV> SpinImageMsg , PrettyPrint &pp
														)
{
    //PrettyPrint pp;
    SITOV msg_in;
    RTOV _rtov;
    _rtov.track_id = track_id;
    _rtov.view_id = view_id;

    for (size_t i = 0; i < SpinImageMsg.size(); i++)
    {
        msg_in = SpinImageMsg.at(i);
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
    ROS_INFO("\t\t[-]v_key: %s, view_id: %i, track_id: %i", v_key.c_str(), view_id, track_id);

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
    vector <  vector <SITOV> > category_instances;
    for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
    {
		vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
        category_instances.push_back(objectViewSpinimages);
    }

    ROS_INFO("OC %s has %d instances", _oc.cat_name.c_str(), _oc.rtov_keys.size());
    //float New_ICD = 0;
    
    intraCategoryDistance(category_instances, _oc.icd, pp);
    //_oc.icd = New_ICD;

    oc_size = ros::serialization::serializationLength(_oc);

    pp.info(std::ostringstream().flush() << _oc.cat_name.c_str() << " category has " << _oc.rtov_keys.size() << " objects.");
    pp.info(std::ostringstream().flush() << "ICD for " << _oc.cat_name.c_str() << " category updated. New ICD is: "<< _oc.icd);

    boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
    PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
    leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
    _pdb->put(oc_key, ocs);

    return (1);
}

void delelteAllRVFromDB ()
{
	vector <string> RVkeys = _pdb->getKeys(key::RV);
	//ROS_INFO("RVs %d exist in the database", RVkeys.size() );
	for (int i = 0; i < RVkeys.size(); i++)
	{
	  //ROS_INFO("delete RTOV = %s", RVkeys.at(i).c_str());
	  _pdb->del(RVkeys.at(i));	  
	}
}

void delelteAllOCFromDB()
{
	vector <string> OCkeys = _pdb->getKeys(key::OC);
	//ROS_INFO("OCs %d exist in the database", OCkeys.size() );
	for (int i = 0; i < OCkeys.size(); i++)
	{
	  //ROS_INFO("delete OC = %s", OCkeys.at(i).c_str());
	  _pdb->del(OCkeys.at(i));
	}
	
}
void delelteAllSITOVFromDB ()
{
	vector <string> SITOVkeys = _pdb->getKeys(key::SI);
	//ROS_INFO("SITOVs %d exist in the database", SITOVkeys.size() );
	for (int i = 0; i < SITOVkeys.size(); i++)
	{
	  //ROS_INFO("delete SITOV = %s", SITOVkeys.at(i).c_str());
	  //delete one TOVI from the db
	  _pdb->del(SITOVkeys.at(i));
	}
}


int deconceptualizingAllTrainData()
{
   delelteAllOCFromDB();
   delelteAllRVFromDB ();
   delelteAllSITOVFromDB ();
  
}


int deleteObjectViewFromSpecificCategory(  std::string cat_name, 
					    unsigned int cat_id, 
					    int track_id,
					    int view_id,
					    vector <SITOV> SpinImageMsg,
					    PrettyPrint &pp)
{

    
    for (size_t i = 0; i < SpinImageMsg.size(); i++)
	{
	   // creat key
	    std::string sp_key = _pdb->makeSIKey(key::SI, track_id, view_id, i );
	    //deleteSpinImage
	    _pdb->del(sp_key); 
	}
// 	
	//create a key for object view
	std::string v_key = _pdb->makeKey(key::RV, track_id, view_id);

	//delete one view from the db
	_pdb->del(v_key);
	
////////////////////////////////////////////////////////////////
	//Update OC with one view
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

	// delete representation of tracked object key from ctegory
		
	if (_oc.rtov_keys.size()==1)
	{	    
	    _oc.rtov_keys.pop_back();
	}
	else
	{
	    ROS_INFO("\t\t[-]- OC_key= %s",oc_key.c_str());
	    for (int i =0 ; i< _oc.rtov_keys.size();i++)
	    {
		if (strcmp(v_key.c_str(),_oc.rtov_keys.at(i).c_str())==0)
		{
// 		    ROS_INFO("\t\t[-]- RTOV key = %s,", _oc.rtov_keys.at(i).c_str());
// 		    ROS_INFO("\t\t[-]- V_key= %s",v_key.c_str());
//     		    ROS_INFO("\t\t[-]- track id = %ld",track_id);
		    string temp = _oc.rtov_keys.at(i).c_str();
		    _oc.rtov_keys.at(i) = _oc.rtov_keys.at(_oc.rtov_keys.size()-1).c_str();
		    _oc.rtov_keys.at(_oc.rtov_keys.size()-1) = temp.c_str();
		    _oc.rtov_keys.pop_back();
		}
	    }
    
	}

	for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
	{
	    ROS_INFO("\t\t[-]- track object view key: %s",_oc.rtov_keys.at(i).c_str());		
	}

	// when an object view delete, ICD should be update for given category
	vector <  vector <SITOV> > category_instances;
	for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
	{		
// 	    ROS_INFO("\t\t[-]- track object view key: %s",_oc.rtov_keys.at(i).c_str());
	    vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
	    category_instances.push_back(objectViewSpinimages);
	}
	
//	float New_ICD = 0.0001;
	intraCategoryDistance(category_instances, _oc.icd, pp);
//	_oc.icd = New_ICD;
		
	oc_size = ros::serialization::serializationLength(_oc);
	
	pp.info(std::ostringstream().flush() << _oc.cat_name.c_str() << " category has " << _oc.rtov_keys.size() << " objects.");
	pp.info(std::ostringstream().flush() << "ICD for " << _oc.cat_name.c_str() << " category updated. New ICD is: "<< _oc.icd);
	
	boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
	PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
	leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
	_pdb->put(oc_key, ocs);

	ROS_INFO("\t\t[-]-Category updated. Size of category after deleting: %ld",category_instances.size());
	return (0);
}

///////////////////////////////////////////////////////////////////
int conceptualizingTrainDataCRC( int &track_id, 
								 PrettyPrint &pp,
						      	 string home_address, 
								 int adaptive_support_lenght,
								 double global_image_width,
								 int threshold,
								 int number_of_bins) 
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string point_cloud_file;

    // read train data from CV_train_instances.txt file line by line
    while (train_data.good ())
    {	
		std::getline (train_data, point_cloud_file);
		if(point_cloud_file.empty () || point_cloud_file.at (0) == '#') // skip blank lines or commented line by #
		{
			continue;
		}

		string pcd_file_path = home_address + point_cloud_file;
		ROS_INFO("\t\t[-]instacne path: %s", pcd_file_path.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		if (io::loadPCDFile <PointT> (pcd_file_path.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]- Cannot read given object %s :", pcd_file_path.c_str());
			return(0);
		}
		
	    // SITOV is a ros msg that is created in race_perception_msgs package. 
		// SITOV stands for Spin Image of Tracked Object View 
		SITOV object_representation;

		/* ______________________________________________
		|                                                |
		|  Compute GOOD descriptor for the given object  |
		|________________________________________________| */

		// boost::shared_ptr<pcl::PointCloud<PointT> > pca_object_view (new PointCloud<PointT>);
		// boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
		// vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
		// double largest_side = 0;
		// int  sign = 1;
		// vector <float> view_point_entropy;
		// string std_name_of_sorted_projected_plane;

		// Eigen::Vector3f center_of_bbox;

		// vector< float > object_description;
    	
		// compuet_object_description( target_pc,
		// 							adaptive_support_lenght,
		// 							global_image_width,
		// 							threshold,
		// 							number_of_bins,
		// 							pca_object_view,
		// 							center_of_bbox,
		// 							vector_of_projected_views, 
		// 							largest_side, 
		// 							sign,
		// 							view_point_entropy,
		// 							std_name_of_sorted_projected_plane,
		// 							object_description );

		// // fill object_representation.spin_image by the elements of object_description
		// for (size_t i = 0; i < object_description.size(); i++)
		// {
		// 	object_representation.spin_image.push_back(object_description.at(i));
		// }


		/* __________________________________________________________
		|                                                           |
		|  Compute the ESF shape description for given point cloud  |
		|___________________________________________________________| */

		pcl::PointCloud<pcl::ESFSignature640>::Ptr esf (new pcl::PointCloud<pcl::ESFSignature640> ());
		estimateESFDescription (target_pc, esf);

		size_t esf_size = sizeof(esf->points.at(0).histogram)/sizeof(float);
		for (size_t i = 0; i < esf_size ; i++)
		{
			object_representation.spin_image.push_back( esf->points.at(0).histogram[i]);
		}

		ROS_INFO("\t\t[-]given object is represented by a histogram with %ld elements",object_representation.spin_image.size());
		
		//// add the representation of given object into a specific category
		std::string category_name;
		category_name = extractCategoryName(point_cloud_file);
		addObjectViewHistogramInSpecificCategory(category_name, 1, track_id, 1, object_representation , pp);	
		track_id++;
	}
	return (1);
}
 


int deleteObjectViewHistogramFromSpecificCategory(std::string cat_name, unsigned int cat_id, 
						    int track_id,  int view_id,
						    PrettyPrint &pp)
{
	
    // creat key
    std::string sp_key = _pdb->makeSIKey(key::SI, track_id, view_id, 1 );
    //deleteSpinImage
    _pdb->del(sp_key); 
	
    //create a key for object view
    std::string v_key = _pdb->makeKey(key::RV, track_id, view_id);
    
    //delete one view from the db
    _pdb->del(v_key);
    	
	////////////////////////////////////////////////////////////////
	//Update OC with one view
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
	ROS_INFO("\t\t TEST4");

	// delete representation of tracked object key from ctegory
	
	if (_oc.rtov_keys.size()==1)
	{	    
	    _oc.rtov_keys.pop_back();
	}
	else
	{
	    ROS_INFO("\t\t[-]- OC_key= %s",oc_key.c_str());
	    for (int i =0 ; i< _oc.rtov_keys.size();i++)
	    {
		if (strcmp(v_key.c_str(),_oc.rtov_keys.at(i).c_str())==0)
		{
// 		    ROS_INFO("\t\t[-]- RTOV key = %s,", _oc.rtov_keys.at(i).c_str());
// 		    ROS_INFO("\t\t[-]- V_key= %s",v_key.c_str());
//     		    ROS_INFO("\t\t[-]- track id = %ld",track_id);
		    string temp = _oc.rtov_keys.at(i).c_str();
		    _oc.rtov_keys.at(i) = _oc.rtov_keys.at(_oc.rtov_keys.size()-1).c_str();
		    _oc.rtov_keys.at(_oc.rtov_keys.size()-1) = temp.c_str();
		    _oc.rtov_keys.pop_back();
		}
	    }
    
	}

	for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
	{
	    ROS_INFO("\t\t[-]- track object view key: %s",_oc.rtov_keys.at(i).c_str());		
	}

	// when an object view delete, ICD should be update for given category
	vector <  vector <SITOV> > category_instances;
	for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
	{		
// 	    ROS_INFO("\t\t[-]- track object view key: %s",_oc.rtov_keys.at(i).c_str());
	    vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
	    category_instances.push_back(objectViewSpinimages);
	}
	
	float New_ICD = 0.0001;
// 	intraCategoryDistance(category_instances, New_ICD, pp);
// 	ROS_INFO("\t\t[-]- ICD = %f", New_ICD);
	_oc.icd = New_ICD;
		
	oc_size = ros::serialization::serializationLength(_oc);
	
	pp.info(std::ostringstream().flush() << _oc.cat_name.c_str() << " category has " << _oc.rtov_keys.size() << " objects.");
	pp.info(std::ostringstream().flush() << "ICD for " << _oc.cat_name.c_str() << " category updated. New ICD is: "<< _oc.icd);

	
	boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
	PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
	leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
	_pdb->put(oc_key, ocs);

	ROS_INFO("\t\t[-]-Category updated. Size of category after deleting: %ld",category_instances.size());

	return (0);
}

//
int deleteAnSpecificObjectView(std::string cat_name, unsigned int cat_id, 
				int track_id,  int view_id,
				boost::shared_ptr< vector <SITOV> > SpinImageMsg)
{
	
    PrettyPrint pp ;
    for (size_t i = 0; i < SpinImageMsg->size(); i++)
    {
	// creat key
	std::string sp_key = _pdb->makeSIKey(key::SI, track_id, view_id, i );
	//deleteSpinImage
	_pdb->del(sp_key); 
    }
    
    
    //create a key for object view
    std::string v_key = _pdb->makeKey(key::RV, track_id, view_id);
    // ROS_INFO("\t\t[-]v_key: %s, view_id: %i, track_id: %i", v_key.c_str(), view_id, track_id);

    //delete one view from the db
    _pdb->del(v_key);
	
////////////////////////////////////////////////////////////////
	//Update OC with one view
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
	
	for (int i = 0; i<_oc.rtov_keys.size(); i++)
	{
	    _oc.rtov_keys.pop_back();
	}
	
	
	// std:: swap(_oc.rtov_keys.at(view_id),_oc.rtov_keys.end());
	// delete representation of tracked object key from ctegory
/*	
	if (view_id == _oc.rtov_keys.size()-1)
	{	    
	     string temp = _oc.rtov_keys.at(view_id).c_str();
	     _oc.rtov_keys.at(view_id) = _oc.rtov_keys.at(0);
	     _oc.rtov_keys.at(0) = temp.c_str();  
	    _oc.rtov_keys.pop_back();
	}
	else
	{
	    string temp = _oc.rtov_keys.at(view_id).c_str();
	    _oc.rtov_keys.at(view_id) = _oc.rtov_keys.at(_oc.rtov_keys.size()-1).c_str();
	    _oc.rtov_keys.at(_oc.rtov_keys.size()-1) = temp.c_str();
	    _oc.rtov_keys.pop_back();
	}
	*/
	ROS_INFO("\t\t[-]- category size after deleting: %ld",_oc.rtov_keys.size());
	for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
	{
	    ROS_INFO("\t\t[-]- track object view key: %s",_oc.rtov_keys.at(i).c_str());		
	}

	// when an object view delete, ICD should be update for given category
	vector <  vector <SITOV> > category_instances;
	for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
	{		
// 	    ROS_INFO("\t\t[-]- track object view key: %s",_oc.rtov_keys.at(i).c_str());
	    vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
	    category_instances.push_back(objectViewSpinimages);
	}
	
//	float New_ICD = 0.0001;
	intraCategoryDistance(category_instances, _oc.icd, pp);
//	_oc.icd = New_ICD;
		
	oc_size = ros::serialization::serializationLength(_oc);
	boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
	PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
	leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
	_pdb->put(oc_key, ocs);

	ROS_INFO("\t\t[-]-Category updated. Size of category after deleting: %ld",category_instances.size());

	return (0);
}

// int create_objects_categories (std::string categoriesAddress)
// {
// 	// Load the Categories file that specified in the Category.txt file
// 	std::string path;
// 	path = ros::package::getPath("race_leaveOneOut_evaluation") +"/"+ categoriesAddress.c_str();
// 	ROS_INFO("\t\t[-]-list of category: %s", path.c_str());
	
// 	std::ifstream listOfObjectCategoriesAddress (path.c_str());
// 	std::string categoryAddress;
// 	std::string categoryName;
	    
// 	while (listOfObjectCategoriesAddress.good ())// read categories address
// 	{ 
	    
// 	    std::getline (listOfObjectCategoriesAddress, categoryAddress);
// 	    if(categoryAddress.empty () || categoryAddress.at (0) == '#') // Skip blank lines or comments
// 		continue;
// 	    categoryName=categoryAddress;
// 	    categoryName.resize(13);//Category//AAA (CategoryName)
// 	    ROS_INFO("\t\t[-]-Category name: %s", categoryName.c_str());
// 	    categoryAddress = ros::package::getPath("race_leaveOneOut_evaluation") +"/"+ categoryAddress.c_str();
// 	    // Load the object of a category that specified in the Category.txt file
// 	    std::ifstream categoyInstances;
// 	    categoyInstances.open(categoryAddress.c_str(),std::ifstream::in);
// 	    unsigned int i=0;
// 	    while (categoyInstances.good ())// read instances of a category 
// 	    {
// 		std::string pcd_file_address;
// 		std::getline (categoyInstances, pcd_file_address);
// 		string catAddress = pcd_file_address;
// 		catAddress.resize(13);
		
// 		if(pcd_file_address.empty () || pcd_file_address.at (0) == '#' || catAddress == "Category//Unk") // Skip blank lines or comments
// 		    continue;
		
// 		pcd_file_address = ros::package::getPath("race_leaveOneOut_evaluation") +"/"+ pcd_file_address.c_str();

//  		ROS_INFO("\t\t[-]-Instance: %s", pcd_file_address.c_str());

// 		//load a PCD object  
// // 		ROS_INFO("\t\t[-]-Instance: %s", pcd_file_address.c_str());
// 		boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
// 		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *PCDFile) == -1)
// 		{	
// 			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
// 			return(0);
// 		}
// 		else
// 		{
// // 			ROS_INFO("\t\t[1]-Loaded a point cloud: %s", pcd_file_address.c_str());
// 		}
		
// 		//compute Spin Image 
// 		//Declare a boost share ptr to the SITOV msg
// 		boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
// 		objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
// 		//Call the library function for estimateSpinImages
// 		estimateSpinImages(PCDFile, 
// 				0.01 /*downsampling_voxel_size*/, 
// 				0.05 /*normal_estimation_radius*/,
// 				8    /*spin_image_width*/,
// 				0.0 /*spin_image_cos_angle*/,
// 				1   /*spin_image_minimum_neighbor_density*/,
// 				0.2 /*spin_image_support_lenght*/,
// 				objectViewSpinImages,
// 				30 /*subsample spinimages*/
// 				);
					
// // 		ROS_INFO("\t\t[-]-There are %ld keypoints", SpinImage->size());
// // 		if (SpinImage->size()>0)
// // 		{	
// // 		    ROS_INFO("\t\t[-]-Spin image of keypoint 0 has %ld elements", SpinImage->at(0).spin_image.size());
// // 		}
			  		
// // 	    	putObjectViewSpinImagesInSpecificCategory(categoryName,1,track_id_gloabal,i,objectViewSpinImages,pp);
// 	    	i++;
// 	    	track_id_gloabal++;
// 		ROS_INFO("\t\t[-]-%s created...",categoryName.c_str());
		
// 	    } //while 2
// 	    categoyInstances.close();
// 	    categoyInstances.clear();
	    
// 	}//while1
// 	listOfObjectCategoriesAddress.close();
// 	listOfObjectCategoriesAddress.clear();

// 	return 0;
// }




////////////////////////////////////////////////////////////////////////////////////////////////////

int crossValidationDataCRC(int K_fold, int iteration , string home_address)
{
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    ROS_INFO("\t\t[-]- Category Path = %s", package_path.c_str());
    string test_data_path = package_path + "/CV_test_instances.txt";
    // ROS_INFO("\t\t[-]- test Path = %s", test_data_path.c_str());
    std::ofstream testInstances (test_data_path.c_str(), std::ofstream::trunc);
    string train_data_path = package_path + "/CV_train_instances.txt";
    // ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
    
    std::ofstream trainInstances (train_data_path.c_str(), std::ofstream::trunc);
    
    // string path = home_address+ "/Category/Category.txt";
    string path = home_address + "Category/Category_orginal.txt";
    // string path = home_address+ "Category/Category.txt";
	ROS_INFO("\t\t[-]- database path = %s", path.c_str());

    std::ifstream listOfObjectCategories (path.c_str(), std::ifstream::in);

    int index =0;
    size_t total_number_of_instances = 0;

    while(listOfObjectCategories.good ())
    {
		string categoryAddress;
		std::getline (listOfObjectCategories, categoryAddress);
		if(categoryAddress.empty () || categoryAddress.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}
	
		// string cat_name = categoryAddress.c_str();
		// cat_name.resize(13);
		index ++;
		string category_address = home_address +"/"+ categoryAddress.c_str();
		std::ifstream categoryInstancesTmp (category_address.c_str());
		size_t category_size = 0;
		string pcd_file_address_tmp_tmp;
		string instance_name ;
		while (categoryInstancesTmp.good ())// read instances of a category 
		{
			std::getline (categoryInstancesTmp, pcd_file_address_tmp_tmp);
			if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
			{
				continue;
			}
			instance_name = pcd_file_address_tmp_tmp;
			category_size++;
		}
		categoryInstancesTmp.close();
		string category_name = extractCategoryName(instance_name.c_str());
		ROS_INFO("\t\t[-]- Category %s has %ld object views ", category_name.c_str(), category_size);

		total_number_of_instances += category_size;
		// ROS_INFO("\t\t[-]- Category%i has %ld object views ", index, category_size);
	
		int test_index = int(category_size/K_fold) * (iteration);
		// ROS_INFO("\t\t[-]- test_index = %i", test_index);
		int i = 0;
		std::ifstream categoryInstances (category_address.c_str());
		
		while (categoryInstances.good ())// read instances of a category 
		{
			std::string pcd_file_address;
			std::getline (categoryInstances, pcd_file_address_tmp_tmp);
			if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
			{
				continue;
			}
			
			if (iteration != K_fold-1) 
			{
				if ((i >= test_index) && (i < (int(category_size/K_fold) * (iteration + 1))))
				{
					testInstances << pcd_file_address_tmp_tmp<<"\n";
					// ROS_INFO("\t\t[%ld]- test data added", i);
				}
				else
				{
					if (category_name != "Unknown")
					{
						trainInstances << pcd_file_address_tmp_tmp<<"\n";
						// ROS_INFO("\t\t[%ld]- train data added", i );
					}    
					
				}
			}
			else 
			{
				if (i >= test_index) 
				{
					testInstances << pcd_file_address_tmp_tmp<<"\n";
					// ROS_INFO("\t\t[%ld]- test data added", i);
				}
				else
				{
					if (category_name != "Unknown")
					{
						trainInstances << pcd_file_address_tmp_tmp<<"\n";
						// ROS_INFO("\t\t[%ld]- train data added", i );
					}    	    
					
				}
			}
			i++;
		}
		categoryInstances.close();
    }
    testInstances.close();
    trainInstances.close();

    ROS_INFO("\t\t[-]- total_number_of_instances = %ld ", total_number_of_instances);

    return(1);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

int modelNetTrainTestData(string home_address)
{
	// #include <dirent.h>  //I use dirent.h for get folders in directory which is also available for windows:

    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string test_data_path = package_path + "/CV_test_instances.txt";
    std::ofstream testInstances (test_data_path.c_str(), std::ofstream::trunc);
    string train_data_path = package_path + "/CV_train_instances.txt";
    
    std::ofstream trainInstances (train_data_path.c_str(), std::ofstream::trunc);
    
 	DIR           *d;
  	struct dirent *dir;  
	string dir_path = home_address + "Category/";
  	d = opendir(dir_path.c_str());
  
  	if (d)
  	{
		while ((dir = readdir(d)) != NULL)
		{
			string category_name = dir->d_name;
			//ROS_INFO("file/folder name = %s", strTmp.c_str());

			std::size_t found = category_name.find(".");
			if (found != std::string::npos)
			{
				continue;	
			}

			ROS_INFO("%s is a directory -- folder name should not contain dot [.]", category_name.c_str());
			
			/// train folder
			string dir_path_train = dir_path + category_name.c_str() + "/train/" ;
			ROS_INFO("train folder = %s", dir_path_train.c_str());
			DIR  *category;
			struct dirent *dir_tmp;  

			category = opendir(dir_path_train.c_str());

			if (category)
			{
				while ((dir_tmp = readdir(category)) != NULL)
				{
					string file_name = dir_tmp->d_name;
					std::size_t found = file_name.find(".pcd");
					if (found != std::string::npos)
					{
						// ROS_INFO("%s is a pcd file ", strTmp.c_str());
						//ROS_INFO("file name = %s", file_name.c_str());
						trainInstances <<"Category//" << category_name.c_str() << "//train//" << file_name.c_str()<<"\n";
					}							
				}
			}
	    	closedir(category);

			// Category//bathtub_Category//bathtub_object_1.pcd

			/// test folder
			string dir_path_test = dir_path + category_name.c_str() + "/test/" ;
			ROS_INFO("test folder = %s", dir_path_train.c_str());
			category = opendir(dir_path_test.c_str());

			if (category)
			{
				while ((dir_tmp = readdir(category)) != NULL)
				{
					string file_name = dir_tmp->d_name;
					std::size_t found = file_name.find(".pcd");
					if (found != std::string::npos)
					{
						// ROS_INFO("%s is a pcd file ", strTmp.c_str());
						ROS_INFO("file name = %s", file_name.c_str());
						testInstances <<"Category//" << category_name.c_str() << "//test//" << file_name.c_str()<<"\n";

					}	
						
				}
			}
	    	closedir(category);
		
		} 
	}

    closedir(d);
	testInstances.close();
	trainInstances.close();

    return(1);
}
////////////////////////////////////////////////////////////////////////////////////////////////////

int varingNumberOfCategories(int number_of_categories, string home_address)
{

    string list_of_categories = home_address+ "Category/Category_orginal.txt";;
    std::ofstream new_categories (list_of_categories.c_str(), std::ofstream::trunc);
    
    string path = home_address+ "Category/list_of_categories.txt";
    ROS_INFO("\t\t[-]- database path = %s", path.c_str());

    std::ifstream list_of_object_categories (path.c_str(), std::ifstream::in);

    int i =0;
    while ( (list_of_object_categories.good() ) and (i < number_of_categories) )
    {
		string category_address;
		std::getline (list_of_object_categories, category_address);
		if(category_address.empty () || category_address.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}
		
		new_categories << category_address<<"\n";
		i++;
    }
	
    new_categories.close();
	
    return(1);
}
////////////////////////////////////////////////////////////////////////////////////////////////////



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
    
    ROS_INFO("\t\t[-]- size of object view histogram = %ld", object_representation.spin_image.size());
    
    for (size_t i = 0; i < object_spin_images.size(); i++)
    {		
		SITOV sp1;
		sp1 = object_spin_images.at(i);
		
		float diffrence = 100;
		float diff_temp = 100;
		int id = 0;

		for (size_t j = 0; j < cluster_center.size(); j++)
		{
			SITOV sp2;
			sp2 = cluster_center.at(j);
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
//////////////////////////////////////////////////////////////////////////////////////////////////////
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
 
//////////////////////////////////////////////////////////////////////////////////////////////////////

int keypointSelection( boost::shared_ptr<pcl::PointCloud<PointT> > target_pc, 
						float uniform_sampling_size,
						boost::shared_ptr<pcl::PointCloud<PointT> > uniform_keypoints,
						boost::shared_ptr<pcl::PointCloud<int> > uniform_sampling_indices )
{
	boost::shared_ptr<PointCloud<PointT> > cloud_filtered (new PointCloud<PointT>);
    pcl::VoxelGrid<PointT > voxelized_point_cloud;	
    voxelized_point_cloud.setInputCloud (target_pc);
    voxelized_point_cloud.setLeafSize (uniform_sampling_size, uniform_sampling_size, uniform_sampling_size);
    voxelized_point_cloud.filter (*cloud_filtered);

    //pcl::PointCloud<int> uniform_sampling_indices;
    for (int i = 0; i < cloud_filtered->points.size() ;i++)
    {
		int nearest_point_index = 0;
		double minimum_distance = 1000;
		for (int j = 0; j < target_pc->points.size(); j++)
		{		
			double distance = sqrt( pow((cloud_filtered->points[i].x - target_pc->points[j].x) , 2) +
						pow((cloud_filtered->points[i].y - target_pc->points[j].y) , 2) +
						pow((cloud_filtered->points[i].z - target_pc->points[j].z), 2));
			if (distance < minimum_distance)
			{
				nearest_point_index = j;
				minimum_distance = distance;
			}
		}
		uniform_sampling_indices->push_back(nearest_point_index);
    }

    pcl::copyPointCloud (*target_pc, uniform_sampling_indices->points, *uniform_keypoints);
    return 1;
}

int conceptualizingTrainData(int &track_id, 
			     PrettyPrint &pp,
			     string home_address,
			     int spin_image_width_int,
			     float spin_image_support_lenght_float,
			     size_t subsampled_spin_image_num_keypoints
			    )
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp_tmp);
		if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );
		
		/* ________________________________________________
		|                                                 |
		|  Compute the Spin-Images for given point cloud  |
		|_________________________________________________| */
		//Declare a boost share ptr to the spin image msg
		
		boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
		objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
		
		if (!estimateSpinImages(target_pc, 
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
			return (0);
		}
		pp.info(std::ostringstream().flush() << "Computed " << objectViewSpinImages->size() << " spin images for given point cloud. ");
		
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp_tmp);
		conceptualizeObjectViewSpinImagesInSpecificCategory(categoryName,1,track_id,1,*objectViewSpinImages,pp);
		track_id++;

    }
    return (1);
 }

 
 
 int conceptualizingTrainData2( int &track_id, 
								PrettyPrint &pp,
								string home_address,
								int spin_image_width_int,
								float spin_image_support_lenght_float,
								float uniform_sampling_size )
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp_tmp);
		if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp_tmp;
		ROS_INFO("\t\t[-]- pcd_file_address = %s", pcd_file_address.c_str());

		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc_tmp (new PointCloud<PointT>);
		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc_tmp) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			ROS_INFO("\t\t[-]- Could not read given object");

			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc_tmp->points.size() );
		
		ROS_INFO("\t\t[-]- size of given point cloud  = %d", target_pc_tmp->points.size());
		
		ROS_INFO ("uniform_sampling_size = %f", uniform_sampling_size);


		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		pcl::VoxelGrid<PointT > voxelized_point_cloud;	
		voxelized_point_cloud.setInputCloud (target_pc_tmp);
		voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
		voxelized_point_cloud.filter (*target_pc);
		ROS_INFO( "The size of converted point cloud  = %d ", target_pc->points.size() );

		/* ________________________________________________
		|                                                 |
		|  Compute the Spin-Images for given point cloud  |
		|_________________________________________________| */
		//Declare a boost share ptr to the spin image msg
		
		boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
		objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
		boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
		boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);

		keypointSelection( target_pc, 
							uniform_sampling_size,
							uniform_keypoints,
							uniform_sampling_indices);

				
		ROS_INFO ("number of keypoints = %i", uniform_keypoints->points.size());
		
		// consider only features of kepoints 
		if (!estimateSpinImages2(target_pc, 
								uniform_sampling_size /*downsampling_voxel_size*/, 
								0.05 /*normal_estimation_radius*/,
								spin_image_width_int /*spin_image_width*/,
								0.0 /*spin_image_cos_angle*/,
								1 /*spin_image_minimum_neighbor_density*/,
								spin_image_support_lenght_float /*spin_image_support_lenght*/,
								objectViewSpinImages,
								uniform_sampling_indices /*subsample spinimages*/)
			)
		{
			pp.error(std::ostringstream().flush() << "Could not compute spin images");
			return (0);
		}
		pp.info(std::ostringstream().flush() << "Computed " << objectViewSpinImages->size() << " spin images for given point cloud. ");
		
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp_tmp);
		conceptualizeObjectViewSpinImagesInSpecificCategory(categoryName,1,track_id,1,*objectViewSpinImages,pp);
		track_id++;

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
 

int conceptualizingDictionaryBasedTrainData(int &track_id, 
											PrettyPrint &pp,
											string home_address,
											int spin_image_width_int,
											float spin_image_support_lenght_float,
											double uniform_sampling_size, 
											vector <SITOV> dictionary_of_spin_images
											)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path *** = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp_tmp);
		if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc_tmp (new PointCloud<PointT>);
		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc_tmp) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc_tmp->points.size() );
		
		
		
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		pcl::VoxelGrid<PointT > voxelized_point_cloud;	
		voxelized_point_cloud.setInputCloud (target_pc_tmp);
		voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
		voxelized_point_cloud.filter (*target_pc);
		ROS_INFO( "The size of converted point cloud  = %d ", target_pc->points.size() );
		
		/* ________________________________________________
		|                                                 |
		|  Compute the Spin-Images for given point cloud  |
		|_________________________________________________| */
		//Declare a boost share ptr to the spin image msg
		
		boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
		objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);

		//// consider all features of the object 
		// 	if (!estimateSpinImages(target_pc, 
		// 				0.01 /*downsampling_voxel_size*/, 
		// 				0.05 /*normal_estimation_radius*/,
		// 				spin_image_width_int /*spin_image_width*/,
		// 				0.0 /*spin_image_cos_angle*/,
		// 				1 /*spin_image_minimum_neighbor_density*/,
		// 				spin_image_support_lenght_float /*spin_image_support_lenght*/,
		// 				objectViewSpinImages,
		// 				subsampled_spin_image_num_keypoints /*subsample spinimages*/
		// 	    ))
		// 	{
		// 	    pp.error(std::ostringstream().flush() << "Could not compute spin images");
		// 	    return (0);
		// 	}
		// 	pp.info(std::ostringstream().flush() << "Computed " << objectViewSpinImages->size() << " spin images for given point cloud. ");

		
		boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
		boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
		keypointSelection( target_pc, 
					uniform_sampling_size,
					uniform_keypoints,
					uniform_sampling_indices);
		
		ROS_INFO ("uniform_sampling_size = %f", uniform_sampling_size);
		ROS_INFO ("number of keypoints = %i", uniform_keypoints->points.size());
		
		//// consider only features of kepoints 
		if (!estimateSpinImages2(target_pc, 
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
		
		SITOV object_representation;
		objectRepresentationBagOfWords (dictionary_of_spin_images, *objectViewSpinImages, object_representation);

		ROS_INFO("\nsize of object view histogram %ld",object_representation.spin_image.size());
		
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp_tmp);

		addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
		track_id++;
    }
    return (1);
 }

 
int conceptualizingTrainDataBasedOnGenericAndSpecificDictionaries(int &track_id, 
																	PrettyPrint &pp,
																	string home_address,
																	int spin_image_width_int,
																	float spin_image_support_lenght_float,
																	double uniform_sampling_size, 
																	vector <SITOV> generic_dictionary
																	)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp_tmp);
		if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc_tmp (new PointCloud<PointT>);
		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc_tmp) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc_tmp->points.size() );
				
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		pcl::VoxelGrid<PointT > voxelized_point_cloud;	
		voxelized_point_cloud.setInputCloud (target_pc_tmp);
		voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
		voxelized_point_cloud.filter (*target_pc);
		ROS_INFO( "The size of converted point cloud  = %d ", target_pc->points.size() );
		
		/* ________________________________________________
		|                                                 |
		|  Compute the Spin-Images for given point cloud  |
		|_________________________________________________| */
		//Declare a boost share ptr to the spin image msg
		
		boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
		objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);

		// 	if (!estimateSpinImages(target_pc, 
		// 				0.01 /*downsampling_voxel_size*/, 
		// 				0.05 /*normal_estimation_radius*/,
		// 				spin_image_width_int /*spin_image_width*/,
		// 				0.0 /*spin_image_cos_angle*/,
		// 				1 /*spin_image_minimum_neighbor_density*/,
		// 				spin_image_support_lenght_float /*spin_image_support_lenght*/,
		// 				objectViewSpinImages,
		// 				subsampled_spin_image_num_keypoints /*subsample spinimages*/
		// 	    ))
		// 	{
		// 	    pp.error(std::ostringstream().flush() << "Could not compute spin images");
		// 	    return (0);
		// 	}
		// 	pp.info(std::ostringstream().flush() << "Computed " << objectViewSpinImages->size() << " spin images for given point cloud. ");

		
		boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
		boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
		keypointSelection( target_pc, 
					uniform_sampling_size,
					uniform_keypoints,
					uniform_sampling_indices);
		
		ROS_INFO ("uniform_sampling_size = %f", uniform_sampling_size);
		ROS_INFO ("number of keypoints = %i", uniform_keypoints->points.size());
		
		
		if (!estimateSpinImages2(target_pc, 
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
		
		
		
		///generic object representation
		SITOV generic_object_representation;
		objectRepresentationBagOfWords (generic_dictionary, *objectViewSpinImages, generic_object_representation);

		ROS_INFO("\nsize of object view histogram %ld",generic_object_representation.spin_image.size());
		
		///specific object representation
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp_tmp);
		
		
		vector <SITOV> specific_dictionary ;
		
		/// clusters7046_Bottle.txt
		char buffer [500];
		double n;
		int spin_image_support_lenght_float_tmp = spin_image_support_lenght_float *100;
		n=sprintf (buffer, "%s%i%i%i%s%s%s","clusters", generic_object_representation.spin_image.size(), spin_image_width_int, spin_image_support_lenght_float_tmp,"_",categoryName.c_str(),".txt");
		string dictionary_name= buffer; 
		ROS_INFO ("dictionary_name = %s", dictionary_name.c_str());
		string dictionary_path = ros::package::getPath("rug_kfold_cross_validation") + "/generic_specific_dictionaries/"+ dictionary_name.c_str();
		ROS_INFO ("dictionary_path = %s", dictionary_path.c_str());
		specific_dictionary = readClusterCenterFromFile (dictionary_path);    
		ROS_INFO ("specific_dictionary_size = %d", specific_dictionary.size());


		///specific object representation
		SITOV specific_object_representation;
		objectRepresentationBagOfWords (specific_dictionary, *objectViewSpinImages, specific_object_representation);

		ROS_INFO("\nsize of object view histogram %ld",specific_object_representation.spin_image.size());


		///concaticating generic and specific representation 
		SITOV object_representation_final;
		object_representation_final.spin_image.insert( object_representation_final.spin_image.end(), generic_object_representation.spin_image.begin(), generic_object_representation.spin_image.end());
		object_representation_final.spin_image.insert( object_representation_final.spin_image.end(), specific_object_representation.spin_image.begin(), specific_object_representation.spin_image.end());

		
		/// concatinating generic and specific dictionaries first and then create object representation
		// 	vector <SITOV> generic_specific_dictionary;
		// 	generic_specific_dictionary.insert( generic_specific_dictionary.end(), generic_dictionary.begin(), generic_dictionary.end());
		// 	generic_specific_dictionary.insert( generic_specific_dictionary.end(), specific_dictionary.begin(), specific_dictionary.end());		      
		// 	
		// 	SITOV object_representation_final;
		// 	objectRepresentationBagOfWords (generic_specific_dictionary, *objectViewSpinImages, object_representation_final);
		// 	ROS_INFO("\nsize of object view histogram %ld",object_representation_final.spin_image.size());
		// 	
		
		///store in memory
		addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation_final , pp);	

		track_id++;
    }
    return (1);
 }
 
 
int deconceptualizingDictionaryBasedTrainData( PrettyPrint &pp,
				string home_address,
				int spin_image_width_int,
				float spin_image_support_lenght_float,
				size_t subsampled_spin_image_num_keypoints )
{
    int track_id = 1;
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp_tmp;
   
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp_tmp);
		if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp_tmp);
		deleteObjectViewHistogramFromSpecificCategory(categoryName, 1, track_id, 1,pp);
		track_id++;
    }
    return (1);
}
 
 
 
int conceptualizingNaiveBayesTrainData2(int &track_id, 
			     PrettyPrint &pp,
			     string home_address,
			     int spin_image_width_int,
			     float spin_image_support_lenght_float,
			     float uniform_sampling_size,
			     int dictionary_size, 
			     vector <SITOV> dictionary
			    )
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
	std::getline (train_data, pcd_file_address_tmp_tmp);
	if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
	{
	    continue;
	}

	string pcd_file_address= home_address + pcd_file_address_tmp_tmp;
	pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
	//load a PCD object   
	boost::shared_ptr<PointCloud<PointT> > target_pc_tmp (new PointCloud<PointT>);
	
	if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc_tmp) == -1)
	{	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
	    return(0);
	}
	pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc_tmp->points.size() );
	
	/* ________________________________________________
	|                                                 |
	|  Compute the Spin-Images for given point cloud  |
	|_________________________________________________| */
	//Declare a boost share ptr to the spin image msg
	
	boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
	objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);

	boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
	pcl::VoxelGrid<PointT > voxelized_point_cloud;	
	voxelized_point_cloud.setInputCloud (target_pc_tmp);
	voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
	voxelized_point_cloud.filter (*target_pc);
	ROS_INFO( "The size of converted point cloud  = %d ", target_pc->points.size() );
	    
	/* ________________________________________________________________________
	|                    		                                    |
	|  Approach 2: voxel approach for keypoints selection and SpinImage    |
	|______________________________________________________________________| */

	boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
	boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
	keypointSelection( target_pc, 
		  uniform_sampling_size,
		  uniform_keypoints,
		  uniform_sampling_indices);

	if (!estimateSpinImages2(target_pc, 
				  0.01 /*downsampling_voxel_size*/, 
				  0.05 /*normal_estimation_radius*/,
				  spin_image_width_int /*spin_image_width*/,
				  0.0 /*spin_image_cos_angle*/,
				  1 /*spin_image_minimum_neighbor_density*/,
				  spin_image_support_lenght_float/*spin_image_support_lenght*/,
				  objectViewSpinImages,
				  uniform_sampling_indices))		    
	{
	    pp.error(std::ostringstream().flush() << "Could not compute spin images");
	    pp.printCallback();
	    return(0);
	}
	
	
	//standard one: 
	//string dictionary_path = ros::package::getPath("race_object_representation") + "/clusters.txt";
	//string dictionary_path = ros::package::getPath("race_object_representation") + "/clusters.txt";
	

	SITOV object_representation;
	notNormalizedObjectRepresentationBagOfWords (dictionary, *objectViewSpinImages, object_representation);

	ROS_INFO("\nsize of object view histogram %ld",object_representation.spin_image.size());
	
	std::string categoryName;
	categoryName = extractCategoryName(pcd_file_address_tmp_tmp);

	addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);

	//debug
	//char ch;
	//ch = getchar();
	
	track_id++;
    }
    return (1);
 }
 
vector <int> generateSequence (int n)
{
    /* initialize random seed: */
    srand (time(NULL));
    vector <int> sequence;
    while( sequence.size() < n )
    {    
		/* generate random number between 1 and n: */
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

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////


int generateRrandomSequencesInstances (string path, string home_address)
{    
    string path1= home_address + path;    
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
    string path2 = home_address + path;
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

int generateRrandomSequencesCategoriesKfold ( string home_address , int number_of_object_per_category)
{

	ROS_INFO("generateRrandomSequencesCategories -- home_address = %s", home_address.c_str());

    std::string path;
    path = home_address +"/Category/Category_orginal.txt";
    std::ifstream listOfObjectCategoriesAddress (path.c_str());

    string categoryAddresstmp = "";
    unsigned int number_of_exist_categories = 0;
    while (listOfObjectCategoriesAddress.good()) 
    {
		std::getline (listOfObjectCategoriesAddress, categoryAddresstmp);
		if(categoryAddresstmp.empty () || categoryAddresstmp.at (0) == '#') // Skip blank lines or comments
			continue;
		number_of_exist_categories++;
    }
    
    vector <int> categories_sequence = generateSequence (number_of_exist_categories);
    std::ofstream categoies;
    string path2 = home_address +"/Category/Category.txt";
    categoies.open (path2.c_str(), std::ofstream::out);
    for (int i =0; i < categories_sequence.size(); i++)
    {
		std::ifstream listOfObjectCategories (path.c_str());
		int j = 0;
		while ((listOfObjectCategories.good()) && (j < categories_sequence.at(i)))
		{
			std::getline (listOfObjectCategories, categoryAddresstmp);
			generateRrandomSequencesInstances(categoryAddresstmp.c_str(), home_address);
			j++;
		}
		categoryAddresstmp.resize(categoryAddresstmp.size()-12);
		categoryAddresstmp+= ".txt";
		categoies << categoryAddresstmp.c_str()<<"\n";
    }
    return 0 ;
  
}


int generateRrandomSequencesCategories ( string home_address , int number_of_object_per_category)
{

    std::string path;
    path = home_address +"/Category/Category_orginal.txt";
    std::ifstream listOfObjectCategoriesAddress (path.c_str());

    string categoryAddresstmp = "";
    unsigned int number_of_exist_categories = 0;
    while (listOfObjectCategoriesAddress.good()) 
    {
		std::getline (listOfObjectCategoriesAddress, categoryAddresstmp);
		if(categoryAddresstmp.empty () || categoryAddresstmp.at (0) == '#') // Skip blank lines or comments
			continue;
		number_of_exist_categories++;
    }
    
    vector <int> categories_sequence = generateSequence (number_of_exist_categories);
    std::ofstream categoies;
    string path2 = home_address +"/Category/Category.txt";
    categoies.open (path2.c_str(), std::ofstream::out);
    for (int i =0; i < categories_sequence.size(); i++)
    {
		std::ifstream listOfObjectCategories (path.c_str());
		int j = 0;
		while ((listOfObjectCategories.good()) && (j < categories_sequence.at(i)))
		{
			std::getline (listOfObjectCategories, categoryAddresstmp);
			generateRrandomSequencesInstances(categoryAddresstmp.c_str(), home_address);
			j++;
		}
		categoryAddresstmp.resize(categoryAddresstmp.size()-12);
		categoryAddresstmp+= ".txt";
		categoies << categoryAddresstmp.c_str()<<"\n";
    }
    return 0 ;
  
}

 
void reportCurrentResults(int TP, int FP, int FN, string fname, bool global)
{

	FILE *result;
	double Precision = TP/double (TP+FP);
	double Recall = TP/double (TP+FN);
	float F1 = 2*(Precision*Recall)/(Precision+Recall);
	
	std::ofstream Result_file;
	Result_file.open (fname.c_str(), std::ofstream::app);
	Result_file.precision(4);
	if (global)
		Result_file << "\n\n\t******************* Global *********************";
	else	
		Result_file << "\n\n\t******************* Lastest run ****************";

	Result_file << "\n\t\t - True  Positive = "<< TP;
	Result_file << "\n\t\t - False Positive = "<< FP;
	Result_file << "\n\t\t - False Negative = "<< FN;
	Result_file << "\n\t\t - Precision  = "<< Precision;
	Result_file << "\n\t\t - Recall = "<< Recall;
	Result_file << "\n\t\t - F-measure = "<< F1;
	Result_file << "\n\n\t************************************************\n\n";
	
	Result_file << "\n------------------------------------------------------------------------------------------------------------------------------------";
	Result_file.close();
	ros::spinOnce();
} 
 
 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int addingGaussianNoise (boost::shared_ptr<PointCloud<PointT> > input_pc, 
			 double standard_deviation,
			 boost::shared_ptr<PointCloud<PointT> > output_pc)

{

      ROS_INFO ("Adding Gaussian noise with mean 0.0 and standard deviation %f\n", standard_deviation);

      *output_pc = *input_pc;

      boost::mt19937 rng; rng.seed (static_cast<unsigned int> (time (0)));
      boost::normal_distribution<> nd (0, standard_deviation);
      boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor (rng, nd);

      for (size_t point_i = 0; point_i < input_pc->points.size (); ++point_i)
      {
		output_pc->points[point_i].x = input_pc->points[point_i].x + static_cast<float> (var_nor ());
		output_pc->points[point_i].y = input_pc->points[point_i].y + static_cast<float> (var_nor ());
		output_pc->points[point_i].z = input_pc->points[point_i].z + static_cast<float> (var_nor ());
      }

    return 0;
 
}

int addingGaussianNoiseXYZL (boost::shared_ptr<PointCloud<pcl::PointXYZL> > input_pc, 
			 double standard_deviation,
			 boost::shared_ptr<PointCloud<pcl::PointXYZL> > output_pc)

{

      ROS_INFO ("Adding Gaussian noise with mean 0.0 and standard deviation %f\n", standard_deviation);

      *output_pc = *input_pc;

      boost::mt19937 rng; rng.seed (static_cast<unsigned int> (time (0)));
      boost::normal_distribution<> nd (0, standard_deviation);
      boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor (rng, nd);

      for (size_t point_i = 0; point_i < input_pc->points.size (); ++point_i)
      {
	output_pc->points[point_i].x = input_pc->points[point_i].x + static_cast<float> (var_nor ());
	output_pc->points[point_i].y = input_pc->points[point_i].y + static_cast<float> (var_nor ());
	output_pc->points[point_i].z = input_pc->points[point_i].z + static_cast<float> (var_nor ());
      }

    return 0;
 
}



int downSamplingXYZL ( boost::shared_ptr<PointCloud<pcl::PointXYZL>  > cloud, 		
		  float downsampling_voxel_size, 
		  boost::shared_ptr<PointCloud<pcl::PointXYZL> > downsampled_pc)
{
  
	// Downsample the input point cloud using downsampling voxel size	
	 
	// Create the filtering object
  	VoxelGrid<PointXYZL > voxel_grid_downsampled_pc;
	voxel_grid_downsampled_pc.setInputCloud (cloud);
	voxel_grid_downsampled_pc.setLeafSize (downsampling_voxel_size, downsampling_voxel_size, downsampling_voxel_size);
	voxel_grid_downsampled_pc.filter (*downsampled_pc);

	
  return 0;
}

int downSampling ( boost::shared_ptr<PointCloud<PointT> > cloud, 		
		  double downsampling_voxel_size, 
		  boost::shared_ptr<PointCloud<PointT> > downsampled_pc)
{
  
	// Downsample the input point cloud using downsampling voxel size	
	 
	// Create the filtering object
  	VoxelGrid<PointT> voxel_grid_downsampled_pc;
	voxel_grid_downsampled_pc.setInputCloud (cloud);
	voxel_grid_downsampled_pc.setLeafSize (downsampling_voxel_size, downsampling_voxel_size, downsampling_voxel_size);
	voxel_grid_downsampled_pc.filter (*downsampled_pc);

	
  return 0;
}

int estimateViewpointFeatureHistogram(boost::shared_ptr<PointCloud<PointT> > cloud, 
				    float normal_estimation_radius,
				    pcl::PointCloud<pcl::VFHSignature308>::Ptr &vfhs)
{ 
// 	const size_t VFH_size = 308;	
// 	ROS_INFO("VFH_size = %ld",VFH_size);

// 	//STEP 1: Downsample the input point cloud using downsampling voxel size
// 	PointCloud<PointT>::Ptr downsampled_pc (new PointCloud<PointT>);
// 	PointCloud<int> sampled_indices;
// 	UniformSampling<PointT> uniform_sampling;
// 	uniform_sampling.setInputCloud (cloud);
// 	uniform_sampling.setRadiusSearch (downsampling_voxel_size/*0.01f*/);
// 	uniform_sampling.compute (sampled_indices);//indices means "fehrest";
// 	copyPointCloud (*cloud, sampled_indices.points, *downsampled_pc);//Keypoints = voxel grid downsampling 
// 	
	//STEP 2: Compute normals for downsampled point cloud
	search::KdTree<PointT>::Ptr kdtree (new search::KdTree<PointT>);
	NormalEstimation<PointT, Normal> normal_estimation;
	normal_estimation.setInputCloud (cloud);
	normal_estimation.setSearchMethod (kdtree);
	normal_estimation.setRadiusSearch ( normal_estimation_radius/*0.05*/);
	PointCloud<Normal>::Ptr normal (new PointCloud< Normal>);
	normal_estimation.compute (*normal);

	//STEP 3: Estimate the VFH for the downsampled_pc with the downsampled_pc_with_normals
	// Create the VFH estimation class, and pass the input dataset+normals to it
	pcl::VFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud (cloud);
	vfh.setInputNormals (normal);
	pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
	vfh.setSearchMethod (tree);
// 	pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());
	vfh.compute (*vfhs);

// 	ROS_INFO("****VFH_size = %ld",vfhs->size ());
	
}


int conceptualizingVFHTrainData( int &track_id, 
				PrettyPrint &pp,
				string home_address, 
				float normal_estimation_radius)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp);
		if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );
		
		/* __________________________________________________________
		|                                                           |
		|  Compute the new shape description for given point cloud  |
		|___________________________________________________________| */

		pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh (new pcl::PointCloud<pcl::VFHSignature308> ());
		// boost::shared_ptr< vector <SITOV> > vfh (new vector <SITOV>);
		estimateViewpointFeatureHistogram(target_pc, 
						normal_estimation_radius,
						vfh);

		size_t vfh_size = sizeof(vfh->points.at(0).histogram)/sizeof(float);
		
		SITOV object_representation;
		for (size_t i = 0; i < vfh_size ; i++)
		{
			object_representation.spin_image.push_back( vfh->points.at(0).histogram[i]);
		}
		//ROS_INFO("VFH_size = %ld",tmp.spin_image.size());
			
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp);

		addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
		track_id++;
    }
    return (1);
 }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int conceptualizingVFHDownSampledTrainData( int &track_id, 
				PrettyPrint &pp,
				string home_address, 
				float normal_estimation_radius,
				float downsampling_voxel_size)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp);
		if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );
		
		//downSampling
		downSampling (  target_pc, 		
						downsampling_voxel_size,
						target_pc);
		
		/* __________________________________________________________
		|                                                           |
		|  Compute the new shape description for given point cloud  |
		|___________________________________________________________| */

		pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh (new pcl::PointCloud<pcl::VFHSignature308> ());
		// boost::shared_ptr< vector <SITOV> > vfh (new vector <SITOV>);
		estimateViewpointFeatureHistogram(target_pc, 
						normal_estimation_radius,
						vfh);

		size_t vfh_size = sizeof(vfh->points.at(0).histogram)/sizeof(float);
		
		SITOV object_representation;
		for (size_t i = 0; i < vfh_size ; i++)
		{
			object_representation.spin_image.push_back( vfh->points.at(0).histogram[i]);
		}
		//ROS_INFO("VFH_size = %ld",tmp.spin_image.size());
			
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp);

		addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
		track_id++;
    }
    return (1);
 }

//////////////////////////////////////////////////////////////////////////////////////
int conceptualizingESFTrainData( int &track_id, 
				  PrettyPrint &pp,
				  string home_address)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
   
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp);
		if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address = home_address + pcd_file_address_tmp;
		//pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		if (io::loadPCDFile <PointT> (pcd_file_address.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		//pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );
		
		/* __________________________________________________________
		|                                                           |
		|  Compute the ESF shape description for given point cloud  |
		|___________________________________________________________| */

		pcl::PointCloud<pcl::ESFSignature640>::Ptr esf (new pcl::PointCloud<pcl::ESFSignature640> ());
		estimateESFDescription (target_pc, esf);
		size_t esf_size = sizeof(esf->points.at(0).histogram)/sizeof(float);
		SITOV object_representation;
		for (size_t i = 0; i < esf_size ; i++)
		{
			object_representation.spin_image.push_back( esf->points.at(0).histogram[i]);
		}
		ROS_INFO("ESF_size = %ld",object_representation.spin_image.size());

			
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp);

		addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
		track_id++;
    }
    return (1);
 }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int conceptualizingESFDownSampledTrainData( int &track_id, 
					      PrettyPrint &pp,
					      string home_address,
					      float downsampling_voxel_size)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
	std::getline (train_data, pcd_file_address_tmp);
	if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
	{
	    continue;
	}

	string pcd_file_address= home_address + pcd_file_address_tmp;
	pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
	//load a PCD object   
	boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
	if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc) == -1)
	{	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
	    return(0);
	}
	pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );

	//downSampling
	downSampling (  target_pc, 		
			downsampling_voxel_size,
			target_pc);

	/* __________________________________________________________
	|                                                           |
	|  Compute the ESF shape description for given point cloud  |
	|___________________________________________________________| */


	pcl::PointCloud<pcl::ESFSignature640>::Ptr esf (new pcl::PointCloud<pcl::ESFSignature640> ());
	estimateESFDescription (target_pc, esf);
	size_t esf_size = sizeof(esf->points.at(0).histogram)/sizeof(float);
	SITOV object_representation;
	for (size_t i = 0; i < esf_size ; i++)
	{
	    object_representation.spin_image.push_back( esf->points.at(0).histogram[i]);
	}
	//ROS_INFO("ESF_size = %ld",object_representation.spin_image.size());

		
	std::string categoryName;
	categoryName = extractCategoryName(pcd_file_address_tmp);

	addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
	track_id++;
    }
    return (1);
 }
 
////////////////////////////////////////////////////////////////////////////////////// 
int estimateGRSDDescription(boost::shared_ptr<PointCloud<PointT> > cloud, 
				    float normal_estimation_radius,
				    pcl::PointCloud<pcl::GRSDSignature21>::Ptr &grsds)
{

	// Estimate the normals.
	search::KdTree<PointT>::Ptr kdtree (new search::KdTree<PointT>);
	NormalEstimation<PointT, Normal> normal_estimation;
	normal_estimation.setInputCloud (cloud);
	normal_estimation.setSearchMethod (kdtree);
	normal_estimation.setRadiusSearch ( normal_estimation_radius);
	PointCloud<Normal>::Ptr normal (new PointCloud< Normal>);
	normal_estimation.compute (*normal);	
  
	// GRSD estimation object.
	pcl::GRSDEstimation<PointT, pcl::Normal, pcl::GRSDSignature21> grsd;
	grsd.setInputCloud(cloud);
	grsd.setInputNormals(normal);
	grsd.setSearchMethod(kdtree);
	// Search radius, to look for neighbors. Note: the value given here has to be
	// larger than the radius used to estimate the normals.
	grsd.setRadiusSearch(0.05);
	grsd.compute(*grsds);
	
	return 0;
} 
   
//////////////////////////////////////////////////////////////////////////////////////
int conceptualizingGRSDTrainData(   int &track_id, 
									PrettyPrint &pp,		  
									string home_address,
									float normal_estimation_radius )
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
	std::getline (train_data, pcd_file_address_tmp);
	if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
	{
	    continue;
	}

	string pcd_file_address= home_address + pcd_file_address_tmp;
	pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
	//load a PCD object   
	boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
	if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc) == -1)
	{	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
	    return(0);
	}
	pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );


	/* __________________________________________________________
	|                                                            |
	|  Compute the GRSD shape description for given point cloud  |
	|____________________________________________________________| */

	pcl::PointCloud<pcl::GRSDSignature21>::Ptr grsd (new pcl::PointCloud<pcl::GRSDSignature21> ());
	estimateGRSDDescription(target_pc, normal_estimation_radius, grsd);
	
	size_t grsd_size = sizeof(grsd->points.at(0).histogram)/sizeof(float);
	SITOV object_representation;
	for (size_t i = 0; i < grsd_size ; i++)
	{
	    object_representation.spin_image.push_back( grsd->points.at(0).histogram[i]);
	}
	//ROS_INFO("GRSD_size = %ld",object_representation.spin_image.size());

		
	std::string category_name;
	category_name = extractCategoryName(pcd_file_address_tmp);
	addObjectViewHistogramInSpecificCategory(category_name, 1, track_id, 1, object_representation , pp);	
	track_id++;
    }
    return (1);
 }
 
 //////////////////////////////////////////////////////////////////////////////////////
int conceptualizingGRSDDownSampledTrainData( int &track_id, 
											 PrettyPrint &pp,
											 string home_address,
											 float normal_estimation_radius,
											 float downsampling_voxel_size)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
	std::getline (train_data, pcd_file_address_tmp);
	if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
	{
	    continue;
	}

	string pcd_file_address= home_address + pcd_file_address_tmp;
	pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
	//load a PCD object   
	boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
	if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc) == -1)
	{	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
	    return(0);
	}
	pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );

	//downSampling
	downSampling( target_pc, 		
				  downsampling_voxel_size,
				  target_pc);

	/* __________________________________________________________
	|                                                            |
	|  Compute the GRSD shape description for given point cloud  |
	|____________________________________________________________| */

	pcl::PointCloud<pcl::GRSDSignature21>::Ptr grsd (new pcl::PointCloud<pcl::GRSDSignature21> ());
	estimateGRSDDescription(target_pc, normal_estimation_radius, grsd);
	
	size_t grsd_size = sizeof(grsd->points.at(0).histogram)/sizeof(float);
	SITOV object_representation;
	for (size_t i = 0; i < grsd_size ; i++)
	{
	    object_representation.spin_image.push_back( grsd->points.at(0).histogram[i]);
	}
	//ROS_INFO("GRSD_size = %ld",object_representation.spin_image.size());

		
	std::string categoryName;
	categoryName = extractCategoryName(pcd_file_address_tmp);

	addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
	track_id++;
    }
    return (1);
 }
  
////////////////////////////////////////////////////////////////////////////////////// 
int estimateGFPFH(boost::shared_ptr<PointCloud<pcl::PointXYZL> > cloud, 
		   			pcl::PointCloud<pcl::GFPFHSignature16>::Ptr &gfpfhs)
{

	// Note: you should now perform classification on the cloud's points. See the original paper for more details. 
	//For this example, we will now consider 16 different classes, and randomly label each point as one of them.
	for (size_t i = 0; i < cloud->points.size(); ++i)
	{
		cloud->points[i].label = 1 + i % 16;
	}
 
	// GFPFH estimation object.
	pcl::GFPFHEstimation<pcl::PointXYZL, pcl::PointXYZL, pcl::GFPFHSignature16> gfpfh;
	gfpfh.setInputCloud(cloud);
	// Set the object that contains the labels for each point. Thanks to the
	// PointXYZL type, we can use the same object we store the cloud in.
	gfpfh.setInputLabels(cloud);
	// Set the size of the octree leaves to 1cm (cubic).
	gfpfh.setOctreeLeafSize(0.01);
	// Set the number of classes the cloud has been labelled with (default is 16).
	gfpfh.setNumberOfClasses(16);
	gfpfh.compute(*gfpfhs);
	return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int conceptualizingGFPFHTrainData( int &track_id, 
									PrettyPrint &pp,
									string home_address)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
	std::getline (train_data, pcd_file_address_tmp);
	if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
	{
	    continue;
	}

	string pcd_file_address= home_address + pcd_file_address_tmp;
	pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
	//load a PCD object     
	boost::shared_ptr<PointCloud<pcl::PointXYZL> > target_pc (new PointCloud<pcl::PointXYZL>);
	if (io::loadPCDFile <pcl::PointXYZL> (pcd_file_address.c_str(), *target_pc) == -1)
	{	
		ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
		return(0);
	}
	pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );
	
	/* ___________________________________________________________
	|                                                             |
	|  Compute the GFPFH shape description for given point cloud  |
	|_____________________________________________________________| */

	pcl::PointCloud<pcl::GFPFHSignature16>::Ptr gfpfh (new pcl::PointCloud<pcl::GFPFHSignature16> ());
	estimateGFPFH( target_pc, gfpfh);
	
	size_t gfpfh_size = sizeof(gfpfh->points.at(0).histogram)/sizeof(float);
	SITOV object_representation;
	for (size_t i = 0; i < gfpfh_size ; i++)
	{
	    object_representation.spin_image.push_back( gfpfh->points.at(0).histogram[i]);
	}		
	//ROS_INFO("GFPFH_size = %ld",object_representation.spin_image.size());

	std::string categoryName;
	categoryName = extractCategoryName(pcd_file_address_tmp);

	addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
	track_id++;
    }
    return (1);
 }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int conceptualizingGFPFHDownSampledTrainData( int &track_id, 
											  PrettyPrint &pp,
											  string home_address, 
											  float downsampling_voxel_size )
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp);
		if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		
		boost::shared_ptr<PointCloud<pcl::PointXYZL> > target_pc (new PointCloud<pcl::PointXYZL>);
		if (io::loadPCDFile <pcl::PointXYZL> (pcd_file_address.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );
		
		downSamplingXYZL( target_pc, 		
						  downsampling_voxel_size,
						  target_pc);
				
		/* ___________________________________________________________
		|                                                             |
		|  Compute the GFPFH shape description for given point cloud  |
		|_____________________________________________________________| */

		pcl::PointCloud<pcl::GFPFHSignature16>::Ptr gfpfh (new pcl::PointCloud<pcl::GFPFHSignature16> ());
		estimateGFPFH( target_pc, gfpfh);
		
		size_t gfpfh_size = sizeof(gfpfh->points.at(0).histogram)/sizeof(float);
		SITOV object_representation;
		for (size_t i = 0; i < gfpfh_size ; i++)
		{
			object_representation.spin_image.push_back( gfpfh->points.at(0).histogram[i]);
		}		
		//ROS_INFO("GFPFH_size = %ld",object_representation.spin_image.size());

		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp);

		addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
		track_id++;
    }
    return (1);
 }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
													bool modelnet_dataset)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    ofstream train_csv;
	train_csv.open ("/home/hamidreza/Desktop/train.csv", std::ofstream::trunc);

    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp);
		if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		if (io::loadPCDFile <PointT> (pcd_file_address.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		//pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );

		
		//downsampling 0 = false 1 = true
		if (downsampling)
        {
			ROS_INFO("Size of object before downsampling (%f) = %d",downsampling_voxel_size, target_pc->points.size());

            downSampling ( target_pc, 		
			               downsampling_voxel_size,
                           target_pc); 

            ROS_INFO("Size of object after downsampling (%f) = %d",downsampling_voxel_size, target_pc->points.size());   
        }

		/* __________________________________________________________
		|                                                           |
		|  Compute the new shape description for given point cloud  |
		|___________________________________________________________| */

		boost::shared_ptr<pcl::PointCloud<T> > pca_object_view (new PointCloud<PointT>);
		boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
		vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
		double largest_side = 0;
		int  sign = 1;
		vector <float> view_point_entropy;
		string std_name_of_sorted_projected_plane;
		Eigen::Vector3f center_of_bbox (0.0, 0.0, 0.0);
        vector< float > object_description;

		if (modelnet_dataset)	
		{
			///NOTE: since we do not need to compute the PCA of object, we use the for_grasp function
			compuet_object_description_for_grasp ( target_pc,
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


		SITOV object_representation;
		for (size_t i = 0; i < object_description.size(); i++)
		{
			object_representation.spin_image.push_back(object_description.at(i));
		}

		ROS_INFO("\nsize of object view histogram %ld",object_representation.spin_image.size());
		

		SITOV deep_representation_sitov;
		/// call deep learning service to represent the given GOOD description as vgg16  
		race_deep_learning_feature_extraction::deep_representation srv;
		srv.request.good_representation = object_representation.spin_image;
		if (deep_learning_server.call(srv))
		{
			//pp.info(std::ostringstream().flush() << "################ receive server responce with size of " << srv.response.deep_representation.size() );
			ROS_INFO("################ receiving deep learning server response with the size of 128 %ld", srv.response.deep_representation.size() );
			if (srv.response.deep_representation.size() < 1)
				ROS_ERROR("Failed to call service deep learning service");
				
			for (size_t i = 0; i < srv.response.deep_representation.size(); i++)
			{
				deep_representation_sitov.spin_image.push_back(srv.response.deep_representation.at(i));
				train_csv << srv.response.deep_representation.at(i) << ",";
			}
		}
		else
		{
			ROS_ERROR("Failed to call deep learning service");
		}
	
		// char ch;
        // ch = getchar ();

		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp);
		train_csv << categoryName << ",";
		train_csv << "\n";
		
		ros::Time start_time = ros::Time::now();
	   	addObjectViewHistogramInSpecificCategoryDeepLearning(categoryName, 1, track_id, 1, deep_representation_sitov , pp);	
		//ROS_INFO("**** add a new object to PDB took = %f", (ros::Time::now() - start_time).toSec());
		track_id++;
    }
	train_csv.close();
    return (1);
 }
 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int conceptualizingDeepLearningAndGoodDescriptorPlusDataAugmentation( int &track_id, 
																		PrettyPrint &pp,
																		string home_address, 
																		int adaptive_support_lenght,
																		double global_image_width,
																		int threshold,
																		int number_of_bins,
																		ros::ServiceClient deep_learning_server, 
																		bool modelnet_dataset)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp;
    //int track_id =1;
    ofstream train_csv;
	train_csv.open ("/home/hamidreza/Desktop/train.csv", std::ofstream::trunc);

    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp);
		if(pcd_file_address_tmp.empty () || pcd_file_address_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc_original (new PointCloud<PointT>);

		if (io::loadPCDFile <PointT> (pcd_file_address.c_str(), *target_pc_original) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		//pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );

		

		//downsampling 0 = false and 1 = true
		for (double downsampling_voxel_size = 0.001; downsampling_voxel_size < 0.05; downsampling_voxel_size += 0.01)
        {

			ROS_INFO("Size of object before downsampling (%f) = %d",downsampling_voxel_size, target_pc_original->points.size());

			boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
            downSampling ( target_pc_original, 		
			               downsampling_voxel_size,
                           target_pc); 

            ROS_INFO("Size of object after downsampling (%f) = %d",downsampling_voxel_size, target_pc->points.size());   
        

			/* __________________________________________________________
			|                                                           |
			|  Compute the new shape description for given point cloud  |
			|___________________________________________________________| */

			boost::shared_ptr<pcl::PointCloud<T> > pca_object_view (new PointCloud<PointT>);
			boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
			vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
			double largest_side = 0;
			int  sign = 1;
			vector <float> view_point_entropy;
			string std_name_of_sorted_projected_plane;
			Eigen::Vector3f center_of_bbox (0.0, 0.0, 0.0);
			vector< float > object_description;

			if (modelnet_dataset)	
			{
				///NOTE: since we do not need to compute the PCA of object, we use the for_grasp function
				compuet_object_description_for_grasp ( target_pc,
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


			SITOV object_representation;
			for (size_t i = 0; i < object_description.size(); i++)
			{
				object_representation.spin_image.push_back(object_description.at(i));
				
			}

			ROS_INFO("\nsize of object view histogram %ld",object_representation.spin_image.size());
			

			SITOV deep_representation_sitov;
			/// call deep learning service to represent the given GOOD description as vgg16  
			race_deep_learning_feature_extraction::deep_representation srv;
			srv.request.good_representation = object_representation.spin_image;
			if (deep_learning_server.call(srv))
			{
				//pp.info(std::ostringstream().flush() << "################ receive server responce with size of " << srv.response.deep_representation.size() );
				ROS_INFO("################ receive server responce with size of %ld", srv.response.deep_representation.size() );
				if (srv.response.deep_representation.size() < 1)
					ROS_ERROR("Failed to call service deep learning service");
					
				for (size_t i = 0; i < srv.response.deep_representation.size(); i++)
				{
					deep_representation_sitov.spin_image.push_back(srv.response.deep_representation.at(i));
					train_csv << srv.response.deep_representation.at(i) << ",";
				}
			}
			else
			{
				ROS_ERROR("Failed to call deep learning service");
			}
		
			// char ch;
			// ch = getchar ();

			std::string categoryName;
			categoryName = extractCategoryName(pcd_file_address_tmp);
			train_csv << categoryName << ",";
			train_csv << "\n";
			
			ros::Time start_time = ros::Time::now();
			addObjectViewHistogramInSpecificCategoryDeepLearning(categoryName, 1, track_id, 1, deep_representation_sitov , pp);	
			//ROS_INFO("**** add a new object to PDB took = %f", (ros::Time::now() - start_time).toSec());
			track_id++;
		}//loop of downsampling


		//gaussian noise 0 = false and 1 = true
		for (double standard_deviation = 0.001; standard_deviation < 0.05; standard_deviation += 0.01)
        {

			ROS_INFO("Size of object before adding gaussian noise (%f) = %d",standard_deviation, target_pc_original->points.size());

			boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
           
		   addingGaussianNoise (target_pc_original,
			   					standard_deviation,
								target_pc);
		          

			/* __________________________________________________________
			|                                                           |
			|  Compute the new shape description for given point cloud  |
			|___________________________________________________________| */

			boost::shared_ptr<pcl::PointCloud<T> > pca_object_view (new PointCloud<PointT>);
			boost::shared_ptr<PointCloud<PointT> > pca_pc (new PointCloud<PointT>); 
			vector < boost::shared_ptr<pcl::PointCloud<PointT> > > vector_of_projected_views;
			double largest_side = 0;
			int  sign = 1;
			vector <float> view_point_entropy;
			string std_name_of_sorted_projected_plane;
			Eigen::Vector3f center_of_bbox (0.0, 0.0, 0.0);
			vector< float > object_description;

			if (modelnet_dataset)	
			{
				///NOTE: since we do not need to compute the PCA of object, we use the for_grasp function
				compuet_object_description_for_grasp ( target_pc,
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


			SITOV object_representation;
			for (size_t i = 0; i < object_description.size(); i++)
			{
				object_representation.spin_image.push_back(object_description.at(i));
				
			}

			ROS_INFO("\nsize of object view histogram %ld",object_representation.spin_image.size());
			

			SITOV deep_representation_sitov;
			/// call deep learning service to represent the given GOOD description as vgg16  
			race_deep_learning_feature_extraction::deep_representation srv;
			srv.request.good_representation = object_representation.spin_image;
			if (deep_learning_server.call(srv))
			{
				//pp.info(std::ostringstream().flush() << "################ receive server responce with size of " << srv.response.deep_representation.size() );
				ROS_INFO("################ receive server responce with size of %ld", srv.response.deep_representation.size() );
				if (srv.response.deep_representation.size() < 1)
					ROS_ERROR("Failed to call service deep learning service");
					
				for (size_t i = 0; i < srv.response.deep_representation.size(); i++)
				{
					deep_representation_sitov.spin_image.push_back(srv.response.deep_representation.at(i));
					train_csv << srv.response.deep_representation.at(i) << ",";
				}
			}
			else
			{
				ROS_ERROR("Failed to call deep learning service");
			}
		
			// char ch;
			// ch = getchar ();

			std::string categoryName;
			categoryName = extractCategoryName(pcd_file_address_tmp);
			train_csv << categoryName << ",";
			train_csv << "\n";
			
			ros::Time start_time = ros::Time::now();
			addObjectViewHistogramInSpecificCategoryDeepLearning(categoryName, 1, track_id, 1, deep_representation_sitov , pp);	
			//ROS_INFO("**** add a new object to PDB took = %f", (ros::Time::now() - start_time).toSec());
			track_id++;
		}//loop of adding noise

    }
	train_csv.close();
    return (1);
 }
 

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int conceptualizingGOODTrainData( int &track_id, 
								  PrettyPrint &pp,
								  string home_address, 
								  int adaptive_support_lenght,
								  double global_image_width,
								  int threshold,
								  int number_of_bins)
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp_tmp);
		if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );
		
		/* __________________________________________________________
		|                                                           |
		|  Compute the new shape description for given point cloud  |
		|___________________________________________________________| */

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
		
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp_tmp);

		addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
		track_id++;
    }
    return (1);
 }
 
////////////////////////////////////////////////////////////////////////////////////////////////////
int conceptualizingGOODDownSampledTrainData( int &track_id, 
								    PrettyPrint &pp,
								    string home_address, 
								    int adaptive_support_lenght,
								    double global_image_width,
								    int threshold,
								    int number_of_bins,
								    float downsampling_voxel_size )
{
   
    string package_path  = ros::package::getPath("rug_kfold_cross_validation");
    string train_data_path = package_path + "/CV_train_instances.txt";
    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string pcd_file_address_tmp_tmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
		std::getline (train_data, pcd_file_address_tmp_tmp);
		if(pcd_file_address_tmp_tmp.empty () || pcd_file_address_tmp_tmp.at (0) == '#') // Skip blank lines or comments
		{
			continue;
		}

		string pcd_file_address= home_address + pcd_file_address_tmp_tmp;
		pp.info(std::ostringstream().flush() << "path: " << pcd_file_address.c_str());
		//load a PCD object   
		boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
		if (io::loadPCDFile <PointXYZRGBA> (pcd_file_address.c_str(), *target_pc) == -1)
		{	
			ROS_ERROR("\t\t[-]-Could not read given object %s :",pcd_file_address.c_str());
			return(0);
		}
		pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc->points.size() );
		
		// downSampling
		downSampling( target_pc, 		
					  downsampling_voxel_size,
					  target_pc);

		
		/* __________________________________________________________
		|                                                           |
		|  Compute the new shape description for given point cloud  |
		|___________________________________________________________| */

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
		
		std::string categoryName;
		categoryName = extractCategoryName(pcd_file_address_tmp_tmp);

		addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation , pp);	
		track_id++;
    }
    return (1);
 }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

void writeToFile (string file_name, float value )
{
    std::ofstream file;
    file.open (file_name.c_str(), std::ofstream::app);
    file.precision(4);
    file << value<<"\n";
    file.close();
    
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool fexists(std::string filename) 
{
  ifstream ifile(filename.c_str());
  return ifile.good();
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int ros_param_set (string topic_name, float value)
{
    char buffer [100];
    double n;
    n=sprintf (buffer, "rosparam set %s %lf",topic_name.c_str(), value);
    string a= buffer; 
    system(a.c_str());
    return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int set_parameters(string name_of_approach)
{
    int number_of_exist_experiments = 0;
    std::string resultsOfExperiments;
    resultsOfExperiments = ros::package::getPath("rug_kfold_cross_validation")+ "/result/results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",resultsOfExperiments.c_str() );
    int exp_num = 0;

    if (!fexists(resultsOfExperiments.c_str()))
    {
	ROS_INFO("File not exist");
	exp_num=1;
    }
    else
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
	exp_num = (number_of_exist_experiments)/2;
	ROS_INFO("number_of_exist_experiments = %i",exp_num );
    }
    
    
    int count = 1;
    double number_of_bins ;
    for (int nb = 10; nb <= 50; nb += 5)
    {
	if (count == exp_num)
	{
	    number_of_bins = nb;
	    ROS_INFO("\t\t[-] number of bins : %i", number_of_bins);    
	    ros_param_set ("/perception/number_of_bins", number_of_bins);
	    break;
	}
	count ++;
    }   
	  
 
    return 0;    
}

////////////////////////////////////////////////////////////////////////////////////////////////////
int numberOfPerformedExperiments(string name_of_approach, int &exp_num)
{

    int number_of_exist_experiments = 0;
    std::string resultsOfExperiments;
    resultsOfExperiments = ros::package::getPath("rug_kfold_cross_validation")+ "/result/results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",resultsOfExperiments.c_str() );

    if (!fexists(resultsOfExperiments.c_str()))
    {
	ROS_INFO("File not exist");
	exp_num=1;
    }
    else
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
	exp_num = (number_of_exist_experiments)/2;
	ROS_INFO("number_of_exist_experiments = %i",exp_num );
    }
        
    return 0;    
}
////////////////////////////////////////////////////////////////////////////////////////////////////
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
											double global_class_accuracy)
{
    //ROS_INFO("TEST report_all_experiments_results fucntion");   
    int number_of_exist_experiments = 0;
    double Precision = TP/double (TP+FP);
    double Recall = TP/double (TP+FN);
    
    double F_measure;
    if ((Precision+Recall)==0)
	F_measure = 10000;//means infinite
    else
	F_measure = (2*Precision*Recall)/double (Precision+Recall);
	
    std::string resultsOfExperiments;
    resultsOfExperiments = ros::package::getPath("rug_kfold_cross_validation")+ "/result/results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",resultsOfExperiments.c_str() );
    int exp_num =1;

    if (!fexists(resultsOfExperiments.c_str()))
    {
		ROS_INFO("File not exist");
		std::ofstream results_of_experiments;
		results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::out);
		results_of_experiments.precision(4);
		results_of_experiments << "EXP"<<"\tnetwork" <<"\t\t\tdataset" <<"\t\t#bins" << "\t\t" << "multiviews" << "\t"<< "normalized" << "\t"<< "pooling" << "\t\t"<< "downsampling" << "\t"<< "Pre"<< "\t"<< "Rec"<< "\t"<< "F1"<< "\t"<< "GCA";
		results_of_experiments << "\n------------------------------------------------------------------------------------------------------------------------------------------------------------------------------";
		if (downsampling)
			results_of_experiments << "\n"<<exp_num << "\t"<< name_of_network<< "\t"<< dataset << "\t" << number_of_bins  <<"\t\t"<< multi_view_flag <<"\t\t"<< image_normalization_flag <<"\t\t"<< pooling_flag <<"\t\t"<< downsampling_voxel_size <<"\t\t"<< Precision<< "\t"<< Recall<< "\t"<< F_measure << "\t"<< global_class_accuracy;
		else
			results_of_experiments << "\n"<<exp_num << "\t"<< name_of_network << "\t"<< dataset << "\t" << number_of_bins  <<"\t\t"<< multi_view_flag <<"\t\t"<< image_normalization_flag <<"\t\t"<< pooling_flag <<"\t\t"<< "FALSE" <<"\t\t"<< Precision<< "\t"<< Recall<< "\t"<< F_measure<< "\t"<< global_class_accuracy;
		results_of_experiments << "\n------------------------------------------------------------------------------------------------------------------------------------------------------------------------------";
		results_of_experiments.close();
		results_of_experiments.clear();
    }
    else
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
		exp_num = (number_of_exist_experiments)/2;
		ROS_INFO("number_of_exist_experiments = %i",exp_num );

		std::ofstream results_of_experiments;
		results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::app);
		results_of_experiments.precision(4);
		if (downsampling)
			results_of_experiments << "\n"<<exp_num << "\t"<< name_of_network<< "\t"<< dataset << "\t" << number_of_bins  <<"\t\t"<< multi_view_flag <<"\t\t"<< image_normalization_flag <<"\t\t"<< pooling_flag <<"\t\t"<< downsampling_voxel_size <<"\t\t"<< Precision<< "\t"<< Recall<< "\t"<< F_measure<< "\t" << global_class_accuracy;
		else
			results_of_experiments << "\n"<<exp_num << "\t"<< name_of_network << "\t"<< dataset << "\t" << number_of_bins  <<"\t\t"<< multi_view_flag <<"\t\t"<< image_normalization_flag <<"\t\t"<< pooling_flag <<"\t\t"<< "FALSE" <<"\t\t"<< Precision<< "\t"<< Recall<< "\t"<< F_measure << "\t"<< global_class_accuracy;
		results_of_experiments << "\n------------------------------------------------------------------------------------------------------------------------------------------------------------------------------";
		results_of_experiments.close();
		results_of_experiments.clear();
		
    } 
   
	return 0;	
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int reportAllExperiments (int TP, int FP, int FN,
							int number_of_bins, 
							string name_of_approach)
{
    //ROS_INFO("TEST reportAllExperiments fucntion");   
    int number_of_exist_experiments = 0;
    double Precision = TP/double (TP+FP);
    double Recall = TP/double (TP+FN);
    
    double F_measure;
    if ((Precision+Recall)==0)
	F_measure = 10000;//means infinite
    else
	F_measure = (2*Precision*Recall)/double (Precision+Recall);

    // int size_of_spin_images = (1+spin_image_width) * (2*spin_image_width+1);
    // ROS_INFO("size_of_spin_images = %i", size_of_spin_images);
    
    // double memory_usage = (size_of_spin_images * 4 * number_of_keypoint)/double(1000);
    // ROS_INFO("memory_usage = %f", memory_usage);
    
    std::string resultsOfExperiments;
    resultsOfExperiments = ros::package::getPath("rug_kfold_cross_validation")+ "/result/results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",resultsOfExperiments.c_str() );
    int exp_num =1;

    if (!fexists(resultsOfExperiments.c_str()))
    {
		ROS_INFO("File not exist");
		std::ofstream results_of_experiments;
		results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::out);
		results_of_experiments.precision(2);
		results_of_experiments << "EXP#"<<"\tnum_of_bins" << "\t"<< "Pre"<< "\t"<< "Rec"<< "\t"<< "F1";
		results_of_experiments << "\n---------------------------------------------------------------------------------";
		results_of_experiments << "\n"<<exp_num<<"\t"<<number_of_bins <<"\t"<< Precision<< "\t"<< Recall<< "\t"<< F_measure;
		results_of_experiments << "\n---------------------------------------------------------------------------------";
		results_of_experiments.close();
		results_of_experiments.clear();
    }
    else
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
		exp_num = (number_of_exist_experiments)/2;
		ROS_INFO("number_of_exist_experiments = %i",exp_num );

		std::ofstream results_of_experiments;
		results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::app);
		results_of_experiments.precision(2);
		results_of_experiments << "\n"<<exp_num<<"\t"<<number_of_bins <<"\t"<< Precision<< "\t"<< Recall<< "\t"<< F_measure;
		results_of_experiments << "\n---------------------------------------------------------------------------------";
		results_of_experiments.close();
		results_of_experiments.clear();
      
    } 
   
return 0;	
}


////////////////////////////////////////////////////////////////////////////////////////////////////
int reportAllExperiments (int TP, int FP, int FN,
			     			string name_of_approach)
{
    //ROS_INFO("TEST reportAllExperiments fucntion");   
    int number_of_exist_experiments = 0;
    double Precision = TP/double (TP+FP);
    double Recall = TP/double (TP+FN);
    
    double F_measure;
    if ((Precision+Recall)==0)
	F_measure = 10000;//means infinite
    else
	F_measure = (2*Precision*Recall)/double (Precision+Recall);

    // int size_of_spin_images = (1+spin_image_width) * (2*spin_image_width+1);
    // ROS_INFO("size_of_spin_images = %i", size_of_spin_images);
    
    // double memory_usage = (size_of_spin_images * 4 * number_of_keypoint)/double(1000);
    // ROS_INFO("memory_usage = %f", memory_usage);
    
    std::string resultsOfExperiments;
    resultsOfExperiments = ros::package::getPath("rug_kfold_cross_validation")+ "/result/results_of_"+name_of_approach+"_experiments.txt";
    ROS_INFO("results of expriments file path = %s",resultsOfExperiments.c_str() );
    int exp_num =1;

    if (!fexists(resultsOfExperiments.c_str()))
    {
		ROS_INFO("File not exist");
		std::ofstream results_of_experiments;
		results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::out);
		results_of_experiments.precision(2);
		results_of_experiments << "EXP"<< "\t"<< "Pre"<< "\t"<< "Rec"<< "\t"<< "F1";
		results_of_experiments << "\n---------------------------------------------------------------------------------";
		results_of_experiments << "\n"<<exp_num<<"\t"<<"\t"<< Precision<< "\t"<< Recall<< "\t"<< F_measure;
		results_of_experiments << "\n---------------------------------------------------------------------------------";
		results_of_experiments.close();
		results_of_experiments.clear();
    }
    else
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
		exp_num = (number_of_exist_experiments)/2;
		ROS_INFO("number_of_exist_experiments = %i",exp_num );

		std::ofstream results_of_experiments;
		results_of_experiments.open (resultsOfExperiments.c_str(), std::ofstream::app);
		results_of_experiments.precision(2);
		results_of_experiments << "\n"<<exp_num<<"\t"<<"\t"<< Precision<< "\t"<< Recall<< "\t"<< F_measure;
		results_of_experiments << "\n---------------------------------------------------------------------------------";
		results_of_experiments.close();
		results_of_experiments.clear();
    } 
   
return 0;	
}

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


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
int jeffreysDiffrenceBetweenTwoObjectViewHistogram(SITOV objectViewHistogram1,
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
			distance_P_Q += (objectViewHistogram1.spin_image.at(i) - objectViewHistogram2.spin_image.at(i)) * log10 (objectViewHistogram2.spin_image.at(i)/objectViewHistogram1.spin_image.at(i));
			distance_P_Q += (objectViewHistogram2.spin_image.at(i) - objectViewHistogram1.spin_image.at(i)) * log10 (objectViewHistogram1.spin_image.at(i)/objectViewHistogram2.spin_image.at(i));
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
int jeffreysBasedObjectCategoryDistance( SITOV target,
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

////////////////////////////////////////////////////////////////////////////////////////////////////
void  confusionMatrixGenerator (  string true_category, string predicted_category, 
                                    std::vector<string> map_category_name_to_index,
                                    std::vector< std::vector <int> > &confusion_matrix )
{
    
    int true_index = -1;
    int predicted_index = -1;
    for (int i = 0; i<confusion_matrix.size(); i++ ) 
    {
        if (true_category.c_str() == map_category_name_to_index.at(i) )
            true_index = i;

        if (predicted_category.c_str() == map_category_name_to_index.at(i) )
            predicted_index = i;
    }

    if ((true_index != -1) && (predicted_index != -1))
    {
        confusion_matrix.at(true_index).at(predicted_index) ++;
    }
    else
    {   
        ROS_INFO("Error computing confusion matrix");
    }
    cout << "confusion_matrix [" << confusion_matrix.size() << "," << confusion_matrix.at(0).size() << "]= \n";
    
    for (int i = 0; i<confusion_matrix.size(); i++ ) 
    { 
		cout << map_category_name_to_index.at(i) << "\t\t";
        for (int j =0; j<confusion_matrix.at(i).size(); j++ ) 
             cout << confusion_matrix.at(i).at(j) << ",\t";
        cout << "\n";
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
void findClosestCategory(vector<float> object_category_distances,
			  int &cat_index, float &mindist, PrettyPrint &pp, float &sigma_distance)
{

	// index = index;
	sigma_distance=0;
	size_t i;
	
	if (object_category_distances.size() < 1)
	{
		//ROS_ERROR("Error: Size of NormalObjectToCategoriesDistances is 0");
		pp.warn(std::ostringstream().flush() << "No categories known");
		cat_index = -1;
		return;
	}
	cat_index = 0;
	mindist = object_category_distances.at(0);
	for (i = 1; i < object_category_distances.size(); i++)
	{
		sigma_distance += object_category_distances.at(i);
		if (mindist > object_category_distances.at(i))
		{
			mindist = object_category_distances.at(i);
			cat_index = i;
		}
	}
	//pp.info(std::ostringstream().flush() << "Sum of distances =" <<sigma_distance << "; numcats=" << object_category_distances.size());
	//pp.info(std::ostringstream().flush() << "min distance =" << mindist);
}


#endif


