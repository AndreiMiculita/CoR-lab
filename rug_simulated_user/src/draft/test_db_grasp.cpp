// ############################################################################
//    
//   Created: 	1/05/2014
//   Author : 	Hamidreza Kasaei
//   Email  :	seyed.hamidreza@ua.pt
//   Purpose: 	add and get Grasp data to the pdb
// 
// ############################################################################

/* _________________________________
   |                                 |
   |          RUN SYSTEM BY          |
   |_________________________________| */
   
//rm -rf /tmp/pdb
//rosrun race_simulated_user test_db_grasp
/* _________________________________
  |                                 |
  |             INCLUDES            |
  |_________________________________| */
  
//ros includes 
#include <ros/ros.h>

//perception db includes
#include <race_perception_msgs/perception_msgs.h>
#include <race_perception_db/perception_db.h>
#include <race_perception_db/perception_db_serializer.h>
#include <race_perception_msgs/GTOV.h>






/* _________________________________
  |                                 |
  |            NameSpace            |
  |_________________________________| */

using namespace std;
using namespace ros;
using namespace race_perception_db;
using namespace race_perception_msgs;


PerceptionDB* _pdb;


/* _________________________________
  |                                 |
  |         Global Variable         |
  |_________________________________| */

    int TID = 0;

				  

// ### Grasp Info
// 
// ### msg: GTOV.msg
// ### Key: GTOV_<TrackID>_<ViewID>
// ### Topic name: GTOV_<TrackID>
// 
// ### Value:
// Header header   ## ROS msg header
// uint32 track_id ##Track ID given by Object Tracking
// uint32 view_id  ##View ID given by Object Tracking
// 
// ### Spin_Image of the grasp point
// float32[] spin_image 
// 
// ### Position of a robot's endefector 
// geometry_msgs/PoseStamped pose_stamped ##The pose of the box frame
// geometry_msgs/Vector3 dimensions ##The dimensions of the box
// 
// ### Finger Position 
// uint32 fingers
// 
// ### Recognition Result
// string object_label  ##category label
// float32 minimum_distance   ##Distance to category

void addGTOVMsg(GTOV _gtove)
{
  
      uint32_t buf_size = ros::serialization::serializationLength(_gtove);
      boost::shared_array<uint8_t> buffer(new uint8_t[buf_size]);
      PerceptionDBSerializer<boost::shared_array<uint8_t>, GTOV>::serialize(buffer, _gtove, buf_size);
      leveldb::Slice slice((char*)buffer.get(), buf_size);
      //string PerceptionDBNodelet::makeKey(const string k_name, const unsigned int tid, const unsigned int vid)
      std::string tovi_key = _pdb->makeKey(key::GTOV, _gtove.track_id, 1 );

      //Put slice to the db
      _pdb->put(tovi_key, slice); 
}
			

GTOV getGTOV (string gtov_key)
{
    GTOV _gtov; 
    string value;
    _pdb->get(gtov_key, &value);
    uint32_t deserial_size = value.length();
    boost::shared_array<uint8_t> buffer(new uint8_t[deserial_size]);
    memcpy(buffer.get(), value.data(), deserial_size);
    race_perception_db::PerceptionDBSerializer<boost::shared_array<uint8_t>, GTOV>::deserialize(buffer, _gtov, deserial_size);

    return ( _gtov );
}

void delelteAllTOVIFromDB ()
{

	vector <string> TOVIkeys = _pdb->getKeys(key::TOVI);
	ROS_INFO("TOVIs %d exist in the database", TOVIkeys.size() );
	for (int i = 0; i < TOVIkeys.size(); i++)
	{
	  //delete one TOVI from the db
	  _pdb->del(TOVIkeys.at(i));
	}
	ROS_INFO("all TOVIs have been deleted (TOVIs %d exist in the database)",TOVIkeys.size() );
}

void timer_callback_GTOV(const ros::TimerEvent& input)
{
    ROS_INFO("timer_callback_GTOV started!");
  

    GTOV gtove;
    gtove.track_id = TID;
    gtove.view_id = 1;
    addGTOVMsg(gtove);
    TID ++;


    //to make an specific key	for an specific
    //std::string gtov_key = _pdb->makeKey(key::GTOV, _gtovi.track_id, 1 );
    
    // to get all exist gtovs keys 
    vector <string> keys = _pdb->getKeys(key::GTOV);
    vector <GTOV> gtovs;

    for ( size_t i = 0; i< keys.size(); ++i)
    {
  
	ROS_INFO( "key: %s", keys.at(i).c_str());
	GTOV _gtov; 
        _gtov = getGTOV (keys.at(i).c_str());
	ROS_INFO ("gtov.track_id (%d)", _gtov.track_id);
        gtovs.push_back(_gtov);

    }

    //pp.info(std::ostringstream().flush() << "rtovs.size(): " << rtovs.size());
    ROS_INFO( "gtovs.size() = %d ", gtovs.size());
    
}



int main(int argc, char** argv)
{
	
    ros::init (argc, argv, "test_db_grasp");
    ros::NodeHandle nh;
    
    // initialize perception database 
    _pdb = race_perception_db::PerceptionDB::getPerceptionDB(&nh); //initialize the database class_list_macros
    string name = nh.getNamespace();

    ros::Timer _timer = nh.createTimer(ros::Duration(1.0), timer_callback_GTOV);

    ros::spin();
    return 0 ;
}


