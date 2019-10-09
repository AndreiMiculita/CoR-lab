// ############################################################################
//    
//   Created: 	1/09/2014
//   Author : 	Hamidreza Kasaei
//   Email  :	seyed.hamidreza@ua.pt
//   Purpose: 	(Dictionary based object representation)
//   		This program follows the teaching protocol and autonomously
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
   |          RUN SYSTEM BY          |
   |_________________________________| */
   
    //rm -rf /tmp/pdb
    //roslaunch race_simulated_user simulated_user_using_dictionary.launch

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

    
    #include <stdlib.h>     /* div, div_t */
    #include <dirent.h>  //I use dirent.h which is also available for windows:
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
  |            constant             |
  |_________________________________| */


 /* _________________________________
  |                                 |
  |        Global Parameters        |
  |_________________________________| */

    //dataset
    std::string home_address;	//  IEETA: 	"/home/hamidreza/";
				//  Washington: "/media/E2480872480847AD/washington/";

    //spin images parameters
    int    spin_image_width = 8;
    double spin_image_support_lenght = 0.06;
    int    subsample_spinimages = 10;
    double recognition_threshold = 100000;  
    double keypoint_sampling_size = 0.01;
    
    
    vector <SITOV> general_dictionary; ///general_dictionary = a set of spin-images like topics 			
    int general_dictionary_size = 90;
    int specific_dictionary_size = 60;
    
    //simulated user parameters
    double P_Threshold = 0.67;  
    int user_sees_no_improvment_const = 100;
    int window_size = 3;
    int number_of_categories =49 ;
    
    ///LDA parameters
    int total_number_of_topic = 50;
    double alpha = 1/*50/total_number_of_topic*/;
    double beta = 0.1;
    int total_number_of_gibbs_sampling_iterations= 20;
								    
    std::string name_of_approach = "ICRA2019";

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

    float F1System =0;

    int TP =0, FP=0, FN=0, TPtmp =0, FPtmp=0, FNtmp=0, Obj_Num=0 , number_of_instances=0 ;

    float PrecisionSystem =0;
    vector <int> recognition_results; // we coded 0: continue(correctly detect unkown object)
				      // 1: TP , 2: FP 
				      //3: FN , 4: FP and FN






/// ------------------------- new functionalities -------------------------
void delelteAllRVFromDB ()
{
    vector <string> RVkeys = _pdb->getKeys(key::RV);
    ROS_INFO("RVs %d exist in the database", RVkeys.size() );
    for (int i = 0; i < RVkeys.size(); i++)
    {
      //ROS_INFO("delete RTOV = %s", RVkeys.at(i).c_str());
      _pdb->del(RVkeys.at(i));	  
    }
    RVkeys = _pdb->getKeys(key::RV);
    ROS_INFO("all RTOVs have been deleted (RTOVs %d exist in the database)", RVkeys.size() );
    
    vector <string> SITOVkeys = _pdb->getKeys(key::SI);
    ROS_INFO("SITOVs %d exist in the database", SITOVkeys.size() );
}

void delelteAllOCFromDB()
{
    vector <string> OCkeys = _pdb->getKeys(key::OC);
    ROS_INFO("OCs %d exist in the database", OCkeys.size() );
    for (int i = 0; i < OCkeys.size(); i++)
    {
      //ROS_INFO("delete OC = %s", OCkeys.at(i).c_str());
      _pdb->del(OCkeys.at(i));
    }
    OCkeys = _pdb->getKeys(key::OC);
    ROS_INFO("all OCs have been deleted (OCs %d exist in the database)",OCkeys.size() );      
}
void delelteAllSITOVFromDB ()
{
    vector <string> SITOVkeys = _pdb->getKeys(key::SI);
    ROS_INFO("SITOVs %d exist in the database", SITOVkeys.size() );
    for (int i = 0; i < SITOVkeys.size(); i++)
    {
      //ROS_INFO("delete SITOV = %s", SITOVkeys.at(i).c_str());
      //delete one TOVI from the db
      _pdb->del(SITOVkeys.at(i));
    }
    SITOVkeys = _pdb->getKeys(key::SI);
    ROS_INFO("all SITOVs have been deleted (SITOVs %d exist in the database)",SITOVkeys.size() );
}


int deconceptualizingAllTrainData()
{
  delelteAllOCFromDB(); 
  ros::Time start_time = ros::Time::now(); 
  while (ros::ok() && (ros::Time::now() - start_time).toSec() <1){ /*wait*/  }	
  delelteAllRVFromDB ();
  start_time = ros::Time::now(); 	
  while (ros::ok() && (ros::Time::now() - start_time).toSec() <1){ /*wait*/  }	
  delelteAllSITOVFromDB ();
  start_time = ros::Time::now(); 
  while (ros::ok() && (ros::Time::now() - start_time).toSec() <1){ /*wait*/  }	

  
}
/////////////////////////////////////////////////////////////////////////////////////////////////////

 int find_nearest_visual_word(SITOV sp1,  vector <SITOV> dictionary, int &dictionary_word_index )
{
      float diffrence=100;
      float diff_temp = 100;
   
      for (size_t i = 0; i < dictionary.size(); i++)
      {
	  SITOV sp2;
	  sp2= dictionary.at(i);
	  if (!differenceBetweenSpinImage(sp1, sp2, diffrence))
	  {	
	      ROS_INFO("\t\t[-]- size of spinimage of dictionary= %ld", sp2.spin_image.size());
	      ROS_INFO("\t\t[-]- size of spinimage of the object= %ld" ,sp1.spin_image.size());
	      ROS_ERROR("Error comparing spin images");
	      return 0;	
	  }
 	  //ROS_INFO("\t\t[-]- diffrence[sp,w%ld] = %f    diff_temp =%f",i, diffrence, diff_temp);
	  if ( diffrence < diff_temp)
	  {
	      diff_temp = diffrence;
	      dictionary_word_index=i;
	  } 
      }
	
 return (1);
} 
/////////////////////////////////////////////////////////////////////////////////////////////////////

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
    _oc.icd = 0;
    _oc.rtov_keys.push_back(v_key);

    // when a new object view add to database, ICD should be update
    vector <  vector <SITOV> > category_instances;
    for (size_t i = 0; i < _oc.rtov_keys.size(); i++)
    {
        vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(_oc.rtov_keys.at(i).c_str());
        category_instances.push_back(objectViewSpinimages);
    }
    
    oc_size = ros::serialization::serializationLength(_oc);

    pp.info(std::ostringstream().flush() << _oc.cat_name.c_str() << " category has " << _oc.rtov_keys.size() << " objects.");
    boost::shared_array<uint8_t> oc_buffer(new uint8_t[oc_size]);
    PerceptionDBSerializer<boost::shared_array<uint8_t>, ObjectCategory>::serialize(oc_buffer, _oc, oc_size);	
    leveldb::Slice ocs((char*)oc_buffer.get(), oc_size);
    _pdb->put(oc_key, ocs);

    return (1);
}


/// estimating the model from scratch
int initialisation( vector <SITOV> dictionary, 
		    int total_number_of_topic,
		    int total_number_of_train_object,
		    vector <int> &total_number_of_words_in_each_object,
		    vector <int> &total_number_of_words_for_topic,
		    vector < vector <int> > &sampled_topic_for_each_word_of_object,
		    vector < vector <int> > &object_topic_matrix,
		    vector < vector <int> > & word_topic_matrix,
		    vector < vector <double> > &theta,
		    vector < vector <double> > &phi,
		    PrettyPrint &pp ) 
{
  
    /// algorithm
    // initialisation
    // zero all count variables, n( m k); nm; n( kt); nk
    // for all documents m 2 [1; M] do
    //   for all words n 2 [1; Nm] in document m do
    //     sample topic index zm;n=k ∼ Mult(1=K)
    //     increment document–topic count: n( m k) + 1
    //     increment document–topic sum: nm + 1
    //     increment topic–term count: n( kt) + 1
    //     increment topic–term sum: nk + 1
    //   end for
    // end for
    
    ///indices and parameters
    //m [1..M] index for objects, 
    //n [1..length of object (M_i)], 
    //w [1..V] index for vocabulary words,
    //k [1..K] index for topics.

    //M total number of object
    //V size of dictionary   
    //T total number of topics

    int M = total_number_of_train_object, V = dictionary.size(), T = total_number_of_topic;
    int m = 0, n = 0, w = 0, k = 0;
        
    /// matrices initialisation
    /// word-topic matrix -> phi  topic_word_matrix (i)(j): number of instances of word/term i assigned to topic j, size V x K
    for (w = 0; w < dictionary.size(); w ++)
    {
      vector<int> row (total_number_of_topic);
      word_topic_matrix.push_back(row);
    }
  
    for (k = 0; k < total_number_of_topic; k ++)
    {
       vector<double> row (dictionary.size());
       phi.push_back(row);
    }
    
    /// object-topic matrix -> theta : object_topic_matrix(i)(j): number of words in document i assigned to topic j, size M x K
    for (m = 0; m < total_number_of_train_object; m ++)
    {
      vector<int> row (total_number_of_topic);
      object_topic_matrix.push_back(row);
      vector<double> row_theta (total_number_of_topic);
      theta.push_back(row_theta);
    }
    
    m = 0;
     
    ROS_INFO("Initial matrices created...");
//     ROS_INFO("word_topic_matrix.size = %ld, word_topic_matrix.at(0).size = %ld", word_topic_matrix.size(),word_topic_matrix.at(0).size());

    /// initialize for random number generation
//     std::srand(std::time(NULL));
    std::srand(std::time(0));

    ///get list of all object categories
    vector <ObjectCategory> ListOfObjectCategory = _pdb->getAllObjectCat();
    ROS_INFO("%ld categories exist in the perception database", ListOfObjectCategory.size() );
    
    for (int a = 0; a < ListOfObjectCategory.size(); a++ )
    { 
      for (int b = 0; b < ListOfObjectCategory.at(a).rtov_keys.size(); b++, m++ )
      { 
	///read spin images of an object view from database
	vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(ListOfObjectCategory.at(a).rtov_keys.at(b).c_str());
	

	vector <int> initial_topics_for_word_i_object_b;
	int x = 0;
	for(int i = 0; i < objectViewSpinimages.size(); i++)
	{
	  /// initialize for topic  
	  int topic = (int)(((double)std::rand() / RAND_MAX) * total_number_of_topic);
// 	  ROS_INFO("generated random number = %d", topic);
 
	  initial_topics_for_word_i_object_b.push_back(topic);	  
	  int w = 0;
	  find_nearest_visual_word(objectViewSpinimages.at(i),dictionary,w);
// 	  ROS_INFO("w = %i", w);

	  /// increase the number of instances of word i assigned to topic j
// 	  ROS_INFO("word_topic_matrix.size = %ld, word_topic_matrix.at(0).size = %ld, topic = %i, w = %i", word_topic_matrix.size(),word_topic_matrix.at(0).size(), topic , w);
	  word_topic_matrix.at(w).at(topic) +=1;
	  /// increase the number of topic assigned to documnet m
//   	  ROS_INFO("object_topic_matrix.size= %ld, m = %i", object_topic_matrix.size(), m);
	  object_topic_matrix.at(m).at(topic) += 1;
	  /// total number of words assigned to topic j
	  total_number_of_words_for_topic.at(topic) += 1;
	}	
	sampled_topic_for_each_word_of_object.push_back(initial_topics_for_word_i_object_b);
	/// total number of words in object b from category a
	total_number_of_words_in_each_object.push_back(objectViewSpinimages.size());
      }
    }
    
    ROS_INFO("Initial topics assigned and phi and theta matrices have been created...");

    return 0;
}


///////////////////////////////////////////////////////
int topicSampling( int m, int n, int w, 
	      float alpha /*constant*/,
	      float beta /*constant*/,
	      vector <SITOV> dictionary, 
	      vector <int> &total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
	      vector <int> &total_number_of_words_in_each_object /*TODO: comput using object_topic_matrix*/,
	      vector < vector <int> > &sampled_topic_for_each_word_of_object,
	      vector < vector <int> > &object_topic_matrix,    
	      vector < vector <int> > &word_topic_matrix,
      	      vector < double > &posterior_distribution_topic,
	      PrettyPrint &pp
	    ) 
{
     
    /// remove topic from the count variables   
    int topic = sampled_topic_for_each_word_of_object.at(m).at(n);
    //ROS_INFO("(m , n , topic )= (%i, %i, %i)", m, n , topic);
    word_topic_matrix.at(w).at(topic) -= 1;
    object_topic_matrix.at(m).at(topic) -= 1;
    total_number_of_words_for_topic.at(topic) -= 1;
    total_number_of_words_in_each_object.at(m) -= 1;


    ///////////////////////////////////////////////////////////////////////////////////
    // 	      dectiption :
    // 	      recall that you should _sample_ a topic from the calculated probability 
    // 	      distribution over topics. This means that every single topic has a 
    // 	      chance to be chosen, the chance being given by its probability in the 
    // 	      distribution. Just choosing the topic with max probability would neglect 
    // 	      the chances of all other topics.
    // 	      The code in both cases draws a random number between 0 and 1 and 
    // 	      determines the topic in the (normalized) topic distribution that belongs 
    // 	      to this number.

    // cumulate multinomial parameters
    // The cumulative distribution function (cdf) is the probability that the variable takes
    // a value less than or equal to x. That is F(x)= Pr[X≤x] = α
    // For a discrete distribution, the cdf can be expressed as
    // F(x) = (∑_(i=0)^x) f(i)
    // http://www.itl.nist.gov/div898/handbook/eda/section3/eda362.htm  (Cumulative Distribution Function)    

    //ROS_INFO ("V = %d", word_topic_matrix.size());
    //ROS_INFO ("K = %d", word_topic_matrix.at(0).size());
    

    double Vbeta = word_topic_matrix.size() * beta;
    double Kalpha = word_topic_matrix.at(0).size() * alpha;    
    
    /// do multinomial sampling via cumulative method
    for (int k = 0; k < word_topic_matrix.at(0).size(); k++) 
    {  
	  double p1 = (word_topic_matrix.at(w).at(k)+ beta)/
		      (total_number_of_words_for_topic.at(k) + Vbeta);
	  // ROS_INFO ("p1 = %f",p1);
	  double p2 = (object_topic_matrix.at(m).at(k)+ alpha)/
		      ((total_number_of_words_in_each_object.at(m) +  Kalpha) -1);
	  
	  posterior_distribution_topic.at (k) = (p1*p2);
    }

    /// cumulate multinomial parameters
    //cout << "posterior_distribution_topic = (";
    for (int k = 1; k < word_topic_matrix.at(0).size(); k++) 
    {
	posterior_distribution_topic.at(k) += posterior_distribution_topic.at(k-1);
	//cout << posterior_distribution_topic.at(k) << ",";
    }
    //cout << ")"<< endl;
   
    /// scaled sample because of unnormalized p[]
    ///a random variable u is sampled from a uniform distribution (draw from Uniform [0,1] : http://www.ics.uci.edu/~newman/pubs/fastlda.pdf)
    
    double u = ((double)std::rand() / (double)(RAND_MAX / posterior_distribution_topic.at(word_topic_matrix.at(0).size()-1)));
    //cout <<"u = " << u << endl;	
    
    ///The final step is to find the interval that u falls into. That is, finding k such that (∑_(z=1)^K-1) P (z|w) < u < (∑_(z=1)^K) P (z|w), where k is the sampled topic of
    ///the current word. Note, that {ρ1 , ρ2 , · · · , ρK } is an increasing sequence, therefore, we just check the u < (∑_(z=1)^K) P (z|w) to find the suitable k (topic).
    
    for (topic = 0; topic < word_topic_matrix.at(0).size()-1; topic++) 
    {
	// ROS_INFO("TEST4");
	 if (posterior_distribution_topic.at(topic) >= u )
	    {
		break;
	    }
    }
    // ROS_INFO("TEST5");
    
    /// add newly estimated topic_i to count variables
    word_topic_matrix.at(w).at(topic) +=1;    
    object_topic_matrix.at(m).at(topic) += 1;
    total_number_of_words_for_topic.at(topic) += 1;
    total_number_of_words_in_each_object.at(m) +=1 ;
    
    //cout <<"smapled topic = " << topic << endl;	
    
    return topic ;
}
////////////////////////////////////////////////////////////////////////////////////////
int computeTheta( float alpha /*constant*/,
		  vector < vector <int> > object_topic_matrix,    
		  vector <int> total_number_of_words_in_each_object,
		  vector < vector <double> > &theta,
		  PrettyPrint &pp) 
{
  /// ROS_INFO ("- \t Inside computing Theta"); 
  for (int m = 0; m < object_topic_matrix.size(); m++) 
  {
    for (int k = 0; k < object_topic_matrix.at(0).size(); k++) 
      {
	theta.at(m).at(k) = (object_topic_matrix.at(m).at(k)+ alpha)/
	  (total_number_of_words_in_each_object.at(m) + object_topic_matrix.at(0).size() * alpha);	    
      }
  }

  return 0;
} 
//////////////////////////////////////////////////////////////////// 
  
int computePhiTrain( float beta /*constant*/,
	        vector <int> total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
		vector < vector <int> > word_topic_matrix,
		vector < vector <double> > &phi,
		PrettyPrint &pp ) 
{
  ROS_INFO ("- \t Inside computing phi"); 
  ROS_INFO (" k -> word_topic_matrix.at(0).size() =%d , w-> word_topic_matrix.size() = %d",word_topic_matrix.at(0).size(), word_topic_matrix.size()); 
  for (int k = 0; k < word_topic_matrix.at(0).size(); k++) 
  {
      for (int w = 0; w < word_topic_matrix.size(); w++) 
      {
	  phi.at(k).at(w) = (word_topic_matrix.at(w).at(k)+ beta)/
	  (total_number_of_words_for_topic.at(k) + word_topic_matrix.size() * beta);
      }
  }
  return 0;
}
////////////////////////////////////////////////////////////////

void saveModel (vector < vector <double> > matrix, 
		  string phi_or_theta,
		  int i )

{ 
  
  char buffer [100];
  double n;
  n=sprintf (buffer, "_%i",i);
  string name = phi_or_theta + buffer;        
	  
  string phat_raw = ros::package::getPath("race_simulated_user")+ "/LDA_models/" +phi_or_theta +"/"+ name + ".txt";
  ofstream matrix_file;
  matrix_file.open (phat_raw.c_str());
//   matrix_file << "clc;\n";    
// //   matrix_file << "close all;\n";
//   matrix_file << "clear all;\n";
//   matrix_file << name+"_fig" <<" = [\n";
  
  for (int i = 0; i < matrix.size(); i++) 
  {
      for (int j = 0; j < matrix.at(0).size(); j++) 
      {
	matrix_file << matrix.at(i).at(j) << " ";
      }
      matrix_file <<"\n";
  }
//   matrix_file << "];\n";
//   matrix_file << "figure();\n";
//   matrix_file << "I = mat2gray("<<name+"_fig"<<");\n";
//   matrix_file << "imshow(1-I);\n";
  matrix_file.close();
  matrix_file.clear();

}
 
////////////////////////////////////////////// 
int estimateModel(int total_number_of_gibbs_sampling_iterations,
		float alpha /*constant*/,
		float beta /*constant*/,
		vector <SITOV> dictionary, 
		vector <int> total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
		vector <int> total_number_of_words_in_each_object /*TODO: comput using object_topic_matrix*/,
		vector < vector <int> > sampled_topic_for_each_word_of_object,
		vector < vector <int> > object_topic_matrix,    
		vector < vector <int> > word_topic_matrix,
		vector < double > &posterior_distribution_topic,
		vector < vector <double> > &theta,
		vector < vector <double> > &phi,
		PrettyPrint &pp )
{
  //TODO : Discripton  
  
  int savestep = 10; /// should be bigger than 1
  ROS_INFO ("- \t totoal number of Gibbs sampling iterations = %d", total_number_of_gibbs_sampling_iterations);

  for (int i = 0; i < total_number_of_gibbs_sampling_iterations; i++)
  {   
        
    /// get list of all object categories
    vector <ObjectCategory> ListOfObjectCategory = _pdb->getAllObjectCat();      
    
    ///for each object do
    int m=0;

    for (int a = 0; a < ListOfObjectCategory.size(); a++ )
    { 
      //ROS_INFO ("category name = %s", ListOfObjectCategory.at(a).cat_name.c_str());
      for (int b = 0; b < ListOfObjectCategory.at(a).rtov_keys.size(); b++, m++ )
      { 
 	  ///read spin images of an object view from database
	  vector <SITOV> objectViewSpinimages = _pdb->getSITOVs(ListOfObjectCategory.at(a).rtov_keys.at(b).c_str());
	  
	  ///for each word do
	  for(int n = 0; n < objectViewSpinimages.size(); n++)
	  {
	      int w = -1;
	      find_nearest_visual_word(objectViewSpinimages.at(n),dictionary,w);
	      //ROS_INFO("W = %i", w);
	      if (w==-1)
	      {
		  w=0;
	      }

	      //ROS_INFO("(m, n, w, objectViewSpinimages.size()) = (%i, %i, %i, %i)", m, n, w , objectViewSpinimages.size());
	      int topic = topicSampling( m, n, w,
					  alpha /*constant*/,
					  beta /*constant*/,
					  dictionary, 
					  total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
					  total_number_of_words_in_each_object /*TODO: comput using object_topic_matrix*/,
					  sampled_topic_for_each_word_of_object,
					  object_topic_matrix,    
					  word_topic_matrix,
					  posterior_distribution_topic,
					  pp) ;

					  
		//ROS_INFO("(m, n, w) = (%i, %i, %i)", m, n, w);		  
		sampled_topic_for_each_word_of_object.at(m).at(n) = topic	;  
		
	  }
      }
      
    }
    
    //TODO put this if as a function for visualizing every certain iteration
    if (i % savestep == 0) 
    {		
	computePhiTrain( beta /*constant*/,
		      total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
		      word_topic_matrix,
		      phi,
		      pp) ;
	  	  
	  saveModel (phi, "phi", i );
	  
	  computeTheta( alpha /*constant*/,
		object_topic_matrix,
		total_number_of_words_in_each_object,    
		theta,
		pp) ;
		
	  saveModel (phi, "theta", i );
	  ROS_INFO ("%ith phi  and theta saved ",i);
    }
    
    
  } /// end of Gibbs sampling

  ROS_INFO ("- \t Gibbs sampling completed!");

  computePhiTrain ( beta /*constant*/,
		  total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
		  word_topic_matrix,
		  phi,
		  pp) ;
		
  computeTheta( alpha /*constant*/,
		object_topic_matrix,
		total_number_of_words_in_each_object,    
		theta,
		pp) ;
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
 int inferenceInitialisation (vector <SITOV> objectViewSpinImages,
		    vector <SITOV> dictionary, 
		    int total_number_of_topic,
		    vector <int> &new_total_number_of_words_in_each_object,
		    vector <int> &new_total_number_of_words_for_topic,
		    vector < vector <int> > &new_sample_topic_index,//TODO: chose a better name of this matrix
		    vector < vector <int> > &new_object_topic_matrix,
		    vector < vector <int> > &new_word_topic_matrix,
		    vector < vector <double> > &new_theta,
		    vector < vector <double> > &new_phi,
		    PrettyPrint &pp) 
 {

    int M = 1, V = dictionary.size(), T = total_number_of_topic;
    int m = 0, n = 0, w = 0, k = 0;
        
    /// matrices initialisation
    /// word-topic matrix -> phi  topic_word_matrix (i)(j): number of instances of word/term i assigned to topic j, size V x K
    for (w = 0; w < dictionary.size(); w ++)
    {
      vector<int> row (total_number_of_topic);
      new_word_topic_matrix.push_back(row);
    }
  
    for (k = 0; k < total_number_of_topic; k ++)
    {
       vector<double> row (dictionary.size());
       new_phi.push_back(row);
    }
    
    /// object-topic matrix -> theta : object_topic_matrix(i)(j): number of words in document i assigned to topic j, size M x K
    for (m = 0; m < 1; m ++)
    {
      vector<int> row (total_number_of_topic);
      new_object_topic_matrix.push_back(row);
      vector<double> row_theta (total_number_of_topic);
      new_theta.push_back(row_theta);
    }
    
    m = 0;
     
    ROS_INFO("inference matrices initialized...");
      
    m = 0;
    vector <int> initial_topics_for_word_i_object_j;
    for(int i = 0; i < objectViewSpinImages.size(); i++)
    {
      /// initialize for topic  
      int topic = (int)(((double)std::rand() / RAND_MAX) * total_number_of_topic);
      initial_topics_for_word_i_object_j.push_back(topic);
      
      ///find nearest word in the dictionary
      int w = 0;
      find_nearest_visual_word(objectViewSpinImages.at(i), dictionary, w);

      /// increase the number of instances of word i assigned to topic j
      new_word_topic_matrix.at(w).at(topic) +=1;
      /// increase the number of topic assigned to documnet m
      new_object_topic_matrix.at(m).at(topic) += 1;
      /// total number of words assigned to topic j
      new_total_number_of_words_for_topic.at(topic) += 1;
      
    }
    ///assigned a topic to each word_i of object_m
    new_sample_topic_index.push_back(initial_topics_for_word_i_object_j);	
    /// total number of words in object i
    new_total_number_of_words_in_each_object.push_back(objectViewSpinImages.size());
 
    return 0;  
}

//////////////////////////////////////////////////////////////////

int inferenceSampling (int m, int n, int w,
			float alpha /*constant*/,
			float beta /*constant*/,
			vector <int> &new_total_number_of_words_in_each_object,
			vector <int> &new_total_number_of_words_for_topic,
			vector < vector <int> > &new_sample_topic_index,//TODO: chose a better name of this matrix
			vector < vector <int> > &new_object_topic_matrix,
			vector < vector <int> > object_topic_matrix,    	
			vector < vector <int> > &new_word_topic_matrix,
			vector < vector <int> > word_topic_matrix				
 		      ) 
{
	
  int topic = new_sample_topic_index.at(m).at(n);
  
  ///decrease the number of instances of word i assigned to topic j
  new_word_topic_matrix.at(w).at(topic) -=1;
  ///decrease the number of topic assigned to documnet m
  new_object_topic_matrix.at(m).at(topic) -= 1;
  ///decrease the total number of words assigned to topic j
  new_total_number_of_words_for_topic.at(topic) -= 1;
  /// decrease the total number of words in object m
  new_total_number_of_words_in_each_object.at(m) -=1 ;
  
  double Vbeta = word_topic_matrix.size() * beta;
  double Kalpha = word_topic_matrix.at(0).size() * alpha; 	
  
  vector < double > posterior_distribution_topic (word_topic_matrix.at(0).size());
  /// do multinomial sampling via cumulative method
  for (int k = 0; k < new_object_topic_matrix.at(0).size(); k++) 
  {  
	double p1 = (word_topic_matrix.at(w).at(k)+ new_word_topic_matrix.at(w).at(k)+ beta)/
		      (new_total_number_of_words_for_topic.at(k) + Vbeta);
		      
	double p2 = (new_object_topic_matrix.at(m).at(k)+ alpha)/ 
	    (new_total_number_of_words_in_each_object.at(m) + Kalpha);

	posterior_distribution_topic.at(k) = (p1*p2);
  }
  
  
  /// cumulate multinomial parameters
  //cout << "posterior_distribution_topic = (";
  for (int k = 1; k < word_topic_matrix.at(0).size(); k++) 
  {
     posterior_distribution_topic.at(k) += posterior_distribution_topic.at(k-1);
    // cout << posterior_distribution_topic.at(k) << ",";
  }
  //cout << ")"<< endl;
  
  /// scaled sample because of unnormalized p[]
  ///a random variable u is sampled from a uniform distribution (draw from Uniform [0,1] : http://www.ics.uci.edu/~newman/pubs/fastlda.pdf)
  
  double u = ((double)std::rand() / (double)(RAND_MAX / posterior_distribution_topic.at(word_topic_matrix.at(0).size()-1)));
  //cout <<"u = " << u << endl;	
  
  ///The final step is to find the interval that u falls into. That is, finding k such that (∑_(z=1)^K-1) P (z|w) < u < (∑_(z=1)^K) P (z|w), where k is the sampled topic of
  ///the current word. Note, that {ρ1 , ρ2 , · · · , ρK } is an increasing sequence, therefore, we just check the u < (∑_(z=1)^K) P (z|w) to find the suitable k (topic).
  
  for (topic = 0; topic < word_topic_matrix.at(0).size(); topic++) 
  {
    if (posterior_distribution_topic.at(topic) > u )
      {
	  break;
      }
  }
    
  /// add newly estimated topic_i to count variables
  new_word_topic_matrix.at(w).at(topic) +=1;    
  new_object_topic_matrix.at(m).at(topic) += 1;
  new_total_number_of_words_for_topic.at(topic) += 1;
  new_total_number_of_words_in_each_object.at(m) +=1 ;
  
  //cout <<"test smapled topic = " << topic << endl;	
  
  return topic;
}

///////////////////////////////////////////////////////////
int computeNewTheta(  float alpha /*constant*/,
		      vector <int> total_number_of_words_in_each_object /*TODO: comput using object_topic_matrix*/,
		      vector < vector <int> > new_object_topic_matrix,    
		      vector < vector <double> > &new_theta,
		      PrettyPrint &pp) 
{
  //size(object_m) = total number of words in object
  //nw[m][t] = number of words in document i assigned to topic j
  //Theta (object-topic_matrix) is computed using following equation:
  //theta[object_i][topic_k] = (nw[m][t] + alpha) / (size(object_m) + K * alpha);
  //reference : Heinrich, Gregor. Parameter estimation for text analysis. Technical report, 2005. 
     
  for (int m = 0; m < new_object_topic_matrix.size(); m++) 
  {
      for (int k = 0; k < new_object_topic_matrix.at(0).size(); k++) 
      {
	  new_theta.at(m).at(k) = (new_object_topic_matrix.at(m).at(k)+ alpha)/
	  (total_number_of_words_in_each_object.at(m) + new_object_topic_matrix.at(0).size() * alpha);
      }
  }

  return 0;
}
/////////////////////////////////////////////////////////////////
int computeNewPhi( float beta /*constant*/,
	        vector <int> total_number_of_words_for_topic /*TODO: compute using object_topic_matrix*/,
	        vector <int> new_total_number_of_words_for_topic /*TODO: compute using object_topic_matrix*/,
		vector < vector <int> > &new_word_topic_matrix,
		vector < vector <int> > &word_topic_matrix,
		vector < vector <double> > &new_phi,
		PrettyPrint &pp ) 
{
  //TODO : write a brif discription and referecne 
  //TODO : chech the size of topic_word_matrix and total_number_of_words_for_topic
  ROS_INFO ("- \t Inside computeNewPhi"); 
  for (int k = 0; k < word_topic_matrix.at(0).size(); k++) 
  {
      for (int w = 0; w < word_topic_matrix.size(); w++) 
      {
	  new_phi.at(k).at(w) = (word_topic_matrix.at(w).at(k)+ new_word_topic_matrix.at(w).at(k)+ beta)/
	  (total_number_of_words_for_topic.at(k) + new_total_number_of_words_for_topic.at(k) + word_topic_matrix.size() * beta);
      }
  }
  return 0;

}
///////////////////////////////////////////////////////////

int inference(int total_number_of_gibbs_sampling_iterations,
		 float alpha /*constant*/,
		 float beta /*constant*/,
		 vector <SITOV> objectViewSpinImages,
		 vector <SITOV> dictionary, 
		 vector < vector <int> > &new_sampled_topic_for_each_word_of_object,//TODO: chose a better name of this matrix
		 vector <int> &new_total_number_of_words_in_each_object,
		 vector <int> total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
		 vector <int> &new_total_number_of_words_for_topic,
 		 vector < vector <int> > object_topic_matrix,    	
		 vector < vector <int> > &new_object_topic_matrix,
		 vector < vector <int> > word_topic_matrix,
		 vector < vector <int> > &new_topic_word_matrix,
		 vector < vector <double> > &new_phi,
		 vector < vector <double> > &new_theta,		 
		 PrettyPrint &pp
		) 
{
 
    ROS_INFO ("- \t totoal number of Gibbs sampling for inference = %d", total_number_of_gibbs_sampling_iterations);
    for (int i = 1; i <= total_number_of_gibbs_sampling_iterations; i++) 
    {
	
      for (int n = 0; n < objectViewSpinImages.size(); n++) 
      {    
	  int w = 0;
	  find_nearest_visual_word(objectViewSpinImages.at(n), dictionary, w);

	  int topic = inferenceSampling(0, n, w,
		  alpha /*constant*/,
		  beta /*constant*/,
		  new_total_number_of_words_in_each_object,
		  new_total_number_of_words_for_topic,
		  new_sampled_topic_for_each_word_of_object,//TODO: chose a better name of this matrix
		  new_object_topic_matrix,
		  object_topic_matrix,    	
		  new_topic_word_matrix,
		  word_topic_matrix);
	  
	  new_sampled_topic_for_each_word_of_object.at(0).at(n) = topic;
      }
    }
    
    vector <int> total_number_of_words_in_each_object;
    total_number_of_words_in_each_object.push_back(objectViewSpinImages.size());
    
    printf("Gibbs sampling for inference completed!\n");
    computeNewTheta( alpha /*constant*/,
		    total_number_of_words_in_each_object /*TODO: comput using object_topic_matrix*/,
		    new_object_topic_matrix,    
		    new_theta,
		    pp);
    
    computeNewPhi(beta /*constant*/,
		  total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
		  new_total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
		  new_topic_word_matrix,
		  word_topic_matrix,
		  new_phi,
		  pp);
    
    
    return 0;
}

/// //////////////////////////////////////////////////////////////////////////////////

int histogramIntersection ( vector <double>  theta,
			    vector <double>  new_theta,
			    double &disimilarity )   
{ 
    disimilarity = 0;
     
    for (int i =0; i< theta.size(); i++)
    {
      
      double tmp = ( new_theta.at(i) > theta.at(i)) ? theta.at(i) : new_theta.at(i);
      disimilarity += tmp;
    }	
    
  return 0;
}

/// //////////////////////////////////////////////////////////////////////////////////

SITOV coefficientofSpinImage (SITOV input_sp, float coefficient)
{
  SITOV output_sp;
  //ROS_INFO ("coefficientofSpinImage");
  for (size_t i = 0; i < input_sp.spin_image.size(); i++)
  {
    output_sp.spin_image.push_back (coefficient * input_sp.spin_image.at(i));  
  }

  return (output_sp);
}

/// //////////////////////////////////////////////////////////////////////////////////

int accumulateTwoSpinImages( SITOV  sp1, SITOV & acc_sps)	
{
  //ROS_INFO ("accumulateTwoSpinImages -> size = %i", acc_sps.spin_image.size());  
  if (acc_sps.spin_image.size() < 1)
  {
      for (size_t i = 0; i < sp1.spin_image.size(); i++)
      {
	  acc_sps.spin_image.push_back( sp1.spin_image.at(i));
      }    
  }
  else 
  {
      //ROS_INFO ("acc_sps.size () = %i , sp1.size() = %i", acc_sps.spin_image.size(), sp1.spin_image.size());  
      for (size_t i = 0; i < sp1.spin_image.size(); i++)
      {
	acc_sps.spin_image.at(i) += sp1.spin_image.at(i);
	  
      }
  }
  return 0;
}

/// //////////////////////////////////////////////////////////////////////////////////

int normalizedSpinImage ( SITOV acc_sps, SITOV & normalized_sp)	
{
    ROS_INFO ("normalizedSpinImage");  

  float sum = 0;
  for (size_t i = 0; i < acc_sps.spin_image.size(); i++)
  {
      sum += acc_sps.spin_image.at(i);
  }
  for (size_t i = 0; i < acc_sps.spin_image.size(); i++)
  {
      normalized_sp.spin_image.push_back( (float) acc_sps.spin_image.at(i) / (float)sum );
  }
  return 0;
}

/// //////////////////////////////////////////////////////////////////////////////////

int createNewDictionaryFromPhi(vector <SITOV> dictionary, 
		    vector < vector <double> > phi,
		    vector <SITOV> &new_dictionary )

{
  ROS_INFO ("k -> phi.size() =%ld, w->phi.at(0).size() = %ld", phi.size(), phi.at(0).size());

  for (int k = 0; k < phi.size(); ++k) 
  {
      SITOV coefficient_word, acc_words, new_normalized_word;
      for (int w = 0; w < phi.at(0).size(); ++w) 
      {
    	    ROS_INFO ("k =%ld, w = %ld", k, w );
			coefficient_word = coefficientofSpinImage(dictionary.at(w), phi.at(k).at(w));
			accumulateTwoSpinImages( coefficient_word, acc_words);
      }
      normalizedSpinImage ( acc_words, new_normalized_word);
      new_dictionary.push_back (new_normalized_word);
  }
  return 0;
}


/// //////////////////////////////////////////////////////////////////////////////////

void  VisualizeNewDictionaryInMatlab ( vector <SITOV> dictionary,   int spin_image_width, double spin_image_support_lenght)
{ 
    string dir_name = ros::package::getPath("race_kfold_cross_validation_evaluation")+ "/generic_specific_dictionaries/learned_topics/";
    string systemStringCommand= "mkdir "+ dir_name;
    system( systemStringCommand.c_str());
    
    DIR *d;
    struct dirent *dir;  
    d = opendir(dir_name.c_str());
    int number_of_file = 0;
    if (d)
    {
      while ((dir = readdir(d)) != NULL)
      {
	  number_of_file ++;
      }
    }
    closedir(d);
    
    char counter [10];
    sprintf( counter, "%d", number_of_file-1 );	
    
    ofstream allFeatures;
    string fileName  =  ros::package::getPath("race_kfold_cross_validation_evaluation") + 
			"/generic_specific_dictionaries/learned_topics/learned_topics_" + counter;
   
    string matlab_file = fileName + ".m";
			
    allFeatures.open (matlab_file.c_str(), std::ofstream::out);
    allFeatures << "% " << "new dictionary based on Topic Modeling, IW = "<< spin_image_width <<", SL = "<<spin_image_support_lenght <<"\n";    
    allFeatures << "clc;\n";    
    allFeatures << "close all;\n";
    allFeatures << "clear all;\n";
    allFeatures << "dictionary = [\n"; 

    for(size_t i = 0; i < dictionary.size(); ++i)
    { 
	for (size_t j = 0; j < dictionary.at( i ).spin_image.size()-1; j++)
	{		
		allFeatures << dictionary.at( i ).spin_image.at( j ) <<","; 
	}
	allFeatures << dictionary.at( i ).spin_image.at( dictionary.at( i ).spin_image.size()-1 ) << "; \n"; 
    }

    allFeatures << "];\n";
    
    string topic_file = fileName + ".txt";
    allFeatures << "%%%%%% write the learned topics to a file to load and use later %%%%%%" ;
    allFeatures << "dlmwrite (" << topic_file<<" , dictionary);";
    
    allFeatures << "%%%%%% visualizing the learned topics %%%%%%" ;
    allFeatures << "figure (1);\n";
    int size_of_dictionary = dictionary.size();
    
    int y = (size_of_dictionary/10);    
    float remain= fmod (size_of_dictionary,10);
    if (remain >0)
    {
	y+=1;
    }
    
    allFeatures << "for i=1 : "<< size_of_dictionary<<"\n";
    allFeatures << "\tA=dictionary(i,:);\n"; 
    allFeatures << "\tB=reshape(A,["<<2*spin_image_width+1<<","<<spin_image_width+1<<"]);\n";
    allFeatures << "\tI = mat2gray(B);\n";
    allFeatures << "\tsubplot(10,"<<y<<",i);\n";
    allFeatures << "\timshow (1-I);\n";
    allFeatures << "\ttitle(strcat('w',int2str(i)));\n";
    allFeatures << "end\n";
    allFeatures << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    
    
    allFeatures << "figure (2);\n"
		<< "j = [20,13,19,79,72,48,51,52,37, 62];%selected topics\n"
		<< "for i=1 : size(j,2)\n"
		<< "\tA=dictionary(j(i),:);\n"
		<< "\tB=reshape(A,[17,9]);\n"
		<< "\tI = mat2gray(B);\n"
		<< "\tsubplot(1,10,i);\n"
		<< "\timshow (I);\n"
		<< "\ttitle(strcat('T',int2str(i)), 'fontsize', 18);\n"
		<< "end\n";

    allFeatures.close();

}

int TrainDataForLDA(  unsigned int &track_id,
		      string home_address,
		      int spin_image_width_int,
		      float spin_image_support_lenght_float,
		      float keypoint_sampling_size,
		      unsigned int &total_number_of_train_object,
		      PrettyPrint &pp )
{
   
    string package_path  = ros::package::getPath("race_kfold_cross_validation_evaluation");
//     string train_data_path = package_path + "/CV_train_instances.txt";
    string train_data_path = package_path + "/CV_test_instances.txt";

    std::ifstream train_data (train_data_path.c_str(), std::ifstream::in);
    ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
 	
    string PCDFileAddressTmp;
    //int track_id =1;
    
    while (train_data.good ())// read train address
    {	
	std::getline (train_data, PCDFileAddressTmp);
	if(PCDFileAddressTmp.empty () || PCDFileAddressTmp.at (0) == '#') // Skip blank lines or comments
	{
	    continue;
	}

	string PCDFileAddress= home_address + PCDFileAddressTmp;
	pp.info(std::ostringstream().flush() << "path: " << PCDFileAddress.c_str());
	//load a PCD object   
	boost::shared_ptr<PointCloud<PointT> > target_pc2 (new PointCloud<PointT>);
	if (io::loadPCDFile <PointXYZRGBA> (PCDFileAddress.c_str(), *target_pc2) == -1)
	{	
	    ROS_ERROR("\t\t[-]-Could not read given object %s :",PCDFileAddress.c_str());
	    return(0);
	}
	pp.info(std::ostringstream().flush() << "The size of given point cloud  = " << target_pc2->points.size() );
	
	boost::shared_ptr<PointCloud<PointT> > target_pc (new PointCloud<PointT>);
	pcl::VoxelGrid<PointT > voxelized_point_cloud;	
	voxelized_point_cloud.setInputCloud (target_pc2);
	voxelized_point_cloud.setLeafSize (0.005, 0.005, 0.005);
	voxelized_point_cloud.filter (*target_pc);
	
// 	/* ________________________________________________
// 	|                                                 |
// 	|  Compute the Spin-Images for given point cloud  |
// 	|_________________________________________________| */
// 	//Declare a boost share ptr to the spin image msg
// 	
// 	boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
// 	objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
// 	
// 	if (!estimateSpinImages(target_pc, 
// 				0.01 /*downsampling_voxel_size*/, 
// 				0.05 /*normal_estimation_radius*/,
// 				spin_image_width_int /*spin_image_width*/,
// 				0.0 /*spin_image_cos_angle*/,
// 				1 /*spin_image_minimum_neighbor_density*/,
// 				spin_image_support_lenght_float /*spin_image_support_lenght*/,
// 				objectViewSpinImages,
// 				subsampled_spin_image_num_keypoints /*subsample spinimages*/))
// 	{
// 	    pp.error(std::ostringstream().flush() << "Could not compute spin images");
// 	    return (0);
// 	}
// 	pp.info(std::ostringstream().flush() << "Computed " << objectViewSpinImages->size() << " spin images for given point cloud. ");

	boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
	objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
	
	boost::shared_ptr<PointCloud<PointT> > uniform_keypoints (new PointCloud<PointT>);
	boost::shared_ptr<pcl::PointCloud<int> >uniform_sampling_indices (new PointCloud<int>);
	keypoint_selection( target_pc, 
			    keypoint_sampling_size,
			    uniform_keypoints,
			    uniform_sampling_indices);
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
	ROS_INFO ( "Computed %ld spin images for given point cloud",  objectViewSpinImages->size() );

	std::string categoryName;
	categoryName = extractCategoryName(PCDFileAddressTmp);
	conceptualizeObjectViewSpinImagesInSpecificCategory(categoryName,1,track_id,1,*objectViewSpinImages,pp);
	track_id++;

    }
    
  /* _____________________________________
    |                                     |
    |   get list of all object categories |
    |_____________________________________| */

    vector <ObjectCategory> ListOfObjectCategory = _pdb->getAllObjectCat();
    //pp.info(std::ostringstream().flush() << ListOfObjectCategory.size()<<" categories exist in the perception database" );
     
    /* _____________________________________________
    |                        	                  |
    |   compute total number of training object  |
    |____________________________________________| */
    
    unsigned int totoal_number_of_training_data =0;
    for (int i = 0; i < ListOfObjectCategory.size(); i++) // all category exist in the database 
    {
	totoal_number_of_training_data += ListOfObjectCategory.at(i).rtov_keys.size();
    }
    
    total_number_of_train_object=totoal_number_of_training_data;
    ROS_INFO("Total number of training data = %d", totoal_number_of_training_data);
    ROS_INFO("track_id = %d", track_id);
    
    return (1);
 }

/////////////////////////////////////////////////////////////////////////////////////////////////////
int IntroduceNewInstanceUsingGeneralSpecificRepresentation ( std::string PCDFileAddress, 
							    unsigned int cat_id, 
							    unsigned int track_id, 
							    unsigned int view_id,
							    PrettyPrint &pp,
							    int spin_image_width_int,
							    float spin_image_support_lenght_float,
							    size_t subsampled_spin_image_num_keypoints,
							    string home_address,
							    vector <SITOV> general_dictionary,
							    int specific_dictionary_size)
{
    string categoryName = extractCategoryName(PCDFileAddress);   
    if(PCDFileAddress.empty () || PCDFileAddress.at (0) == '#' || categoryName == "Category//Unk") // Skip blank lines or comments
    {
	return 0;
    }
    
    PCDFileAddress = home_address +"/"+ PCDFileAddress.c_str();

    ///load a PCD object  
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
			    keypoint_sampling_size,
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
      
    ///generic object representation
    SITOV general_object_representation;
    objectRepresentationBagOfWords (general_dictionary, *objectViewSpinImages, general_object_representation);
        
    ///specific object representation		 
    vector <SITOV> specific_dictionary;
    
    /// clusters7046_Bottle.txt
    string package_path  = ros::package::getPath("race_kfold_cross_validation_evaluation");
    char buffer [500];
    double n;
    int spin_image_support_lenght_float_tmp = spin_image_support_lenght *100;
    n=sprintf (buffer, "%s%i%i%i%s%s%s","clusters", specific_dictionary_size, spin_image_width, spin_image_support_lenght_float_tmp,"_",categoryName.c_str(),".txt");
    string dictionary_name= buffer; 
    ROS_INFO ("dictionary_name = %s", dictionary_name.c_str());
    string IEETA_or_Washington = (home_address == "/home/hamidreza/") ? "IEETA/" : "Washington/";

    string dictionary_path = package_path + "/generic_specific_dictionaries/"+ IEETA_or_Washington.c_str() + dictionary_name.c_str();
    ROS_INFO ("dictionary_path = %s", dictionary_path.c_str());
    specific_dictionary = readClusterCenterFromFile (dictionary_path);    
    ROS_INFO ("specific_dictionary_size = %d", specific_dictionary.size());
		    
    ///specific object representation
    SITOV specific_object_representation;
    objectRepresentationBagOfWords (specific_dictionary, *objectViewSpinImages, specific_object_representation);
    ROS_INFO("\nsize of object view histogram %ld",specific_object_representation.spin_image.size());
        
    ///concaticating generic and specific representation 
    SITOV object_representation_final;
    object_representation_final.spin_image.insert( object_representation_final.spin_image.end(), general_object_representation.spin_image.begin(), general_object_representation.spin_image.end());
    object_representation_final.spin_image.insert( object_representation_final.spin_image.end(), specific_object_representation.spin_image.begin(), specific_object_representation.spin_image.end());

    ROS_INFO("size of object view histogram %ld",object_representation_final.spin_image.size());

    addObjectViewHistogramInSpecificCategory(categoryName, 1, track_id, 1, object_representation_final , pp);
    ROS_INFO("\t\t[-]-%s created...",categoryName.c_str());
        
    return (0);
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int selectAnInstancefromSpecificCategory(unsigned int category_index, 
					 unsigned int &instance_number, 
					 string &Instance,
					 string home_address)
{
    std::string path;
    path =  home_address +"/Category/Category.txt";

    ROS_INFO("TEST");
    ROS_INFO("category index = %d",category_index);
    ROS_INFO("instance_number = %d",instance_number);
   
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
	ROS_INFO("\t\t[-]-The file doesn't exist");
	return -1;
    }
    
    path = home_address +"/"+ categoryAddresstmp.c_str();
    
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
	ROS_INFO("\t\t[-]-The file doesn't exist");
	return -1;
    }
    
    Instance=PCDFileAddressTmp;
    instance_number++;
    return 0;
    
}
////////////////////////////////////////////////////////////////////////////////////////////////////
int introduceNewCategoryUsingGeneralSpecificRepresentation(int class_index,
							  unsigned int &track_id,
							  unsigned int &instance_number,
							  string fname,
							  PrettyPrint &pp,
							  int spin_image_width_int,
							  float spin_image_support_lenght_float,
							  size_t subsampled_spin_image_num_keypoints,
							  string home_address,
							  vector <SITOV> general_dictionary,
							  int specific_dictionary_size)
				  
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
	    IntroduceNewInstanceUsingGeneralSpecificRepresentation ( instance_path, 
								    cat_id, 
								    track_id, 
								    view_id,
								    pp,
								    spin_image_width_int,
								    spin_image_support_lenght_float,
								    subsampled_spin_image_num_keypoints,
								    home_address,
								    general_dictionary,
								    specific_dictionary_size);
	    track_id++;
	    //view_id ++; // in this implementation we consider VID as a constant
	}

	// extracting the category name 
	string categoryName=extractCategoryName(instance_path);
	ROS_INFO("\n extractCategoryName %s", categoryName.c_str()); 
	report_category_introduced(fname,categoryName.c_str());
	return 0;
}

int crossValidationData(int K_fold, int iteration , string home_address)
{
    string package_path  = ros::package::getPath("race_kfold_cross_validation_evaluation");
//     ROS_INFO("\t\t[-]- Category Path = %s", package_path.c_str());
    string test_data_path = package_path + "/CV_test_instances.txt";
//     ROS_INFO("\t\t[-]- test Path = %s", test_data_path.c_str());
    std::ofstream testInstances (test_data_path.c_str(), std::ofstream::trunc);
    string train_data_path = package_path + "/CV_train_instances.txt";
//     ROS_INFO("\t\t[-]- train Path = %s", train_data_path.c_str());
    
    std::ofstream trainInstances (train_data_path.c_str(), std::ofstream::trunc);
    
    string path = home_address+ "/Category/Category.txt";
//     string path = home_address+ "Category/Category_orginal.txt";
//     string path = home_address+ "Category/Category.txt";
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
	
// 	string cat_name = categoryAddress.c_str();
// 	cat_name.resize(13);
	index ++;
	string category_address = home_address +"/"+ categoryAddress.c_str();
	std::ifstream categoryInstancesTmp (category_address.c_str());
	size_t category_size = 0;
	string PCDFileAddressTmp;
	string instance_name ;
	while (categoryInstancesTmp.good ())// read instances of a category 
	{
	    std::getline (categoryInstancesTmp, PCDFileAddressTmp);
	    if(PCDFileAddressTmp.empty () || PCDFileAddressTmp.at (0) == '#') // Skip blank lines or comments
	    {
		continue;
	    }
	    instance_name = PCDFileAddressTmp;
	    category_size++;
	}
	categoryInstancesTmp.close();
	string category_name = extractCategoryName(instance_name.c_str());
	ROS_INFO("\t\t[-]- Category %s has %ld object views ", category_name.c_str(), category_size);

	total_number_of_instances += category_size;
//	ROS_INFO("\t\t[-]- Category%i has %ld object views ", index, category_size);
	
	int test_index = int(category_size/K_fold) * (iteration);
// 	ROS_INFO("\t\t[-]- test_index = %i", test_index);
	int i = 0;
	std::ifstream categoryInstances (category_address.c_str());
    
	while (categoryInstances.good ())// read instances of a category 
	{
	    std::string PCDFileAddress;
	    std::getline (categoryInstances, PCDFileAddressTmp);
	    if(PCDFileAddressTmp.empty () || PCDFileAddressTmp.at (0) == '#') // Skip blank lines or comments
	    {
		continue;
	    }
	    
    	    if (iteration != K_fold-1) 
	    {
		if ((i >= test_index) && (i < (int(category_size/K_fold) * (iteration + 1))))
		{
		    testInstances << PCDFileAddressTmp<<"\n";
// 		    ROS_INFO("\t\t[%ld]- test data added", i);
		}
		else
		{
		    if (category_name != "Unknown")
		    {
			trainInstances << PCDFileAddressTmp<<"\n";
// 			ROS_INFO("\t\t[%ld]- train data added", i );
		    }    
		    
		}
	    }
	    else 
	    {
		if (i >= test_index) 
		{
		    testInstances << PCDFileAddressTmp<<"\n";
// 		    ROS_INFO("\t\t[%ld]- test data added", i);
		}
		else
		{
		    if (category_name != "Unknown")
		    {
			trainInstances << PCDFileAddressTmp<<"\n";
// 			ROS_INFO("\t\t[%ld]- train data added", i );
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

/////////////////////////////////////////////////////////////////////////////////////////////////////

void evaluationfunction(const race_perception_msgs::RRTOV &result)
{
    PrettyPrint pp;
    ROS_INFO("ground_truth_name = %s", result.ground_truth_name.c_str());
    string tmp = result.ground_truth_name.substr(home_address.size(),result.ground_truth_name.size());
    InstancePathTmp = tmp;
    string True_cat = extractCategoryName(tmp);
       
//     string True_cat = True_Category_Global;  
    std:: string Object_name;
    Object_name = extractObjectName (result.ground_truth_name);
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
	IntroduceNewInstanceUsingGeneralSpecificRepresentation ( InstancePathTmp, cat_id, 
								track_id, view_id, pp,
								spin_image_width,
								spin_image_support_lenght,
								subsample_spinimages,
								home_address,
								general_dictionary,
								specific_dictionary_size);
	
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

	IntroduceNewInstanceUsingGeneralSpecificRepresentation ( InstancePathTmp, cat_id, 
								track_id, view_id, pp,
								spin_image_width,
								spin_image_support_lenght,
								subsample_spinimages,
								home_address,
								general_dictionary,
								specific_dictionary_size);
	
	number_of_instances++;
    }
    Result.close();
    Result.clear();
    PrecisionMonitor.close();
    PrecisionMonitor.clear();
    pp.printCallback();
}




int main(int argc, char** argv)
{
  
  
  // 	    string dictionary_path = ros::package::getPath("race_object_representation") + "/clusters.txt";
//     vector <SITOV> cluster_center = readClusterCenterFromFile (dictionary_path);
//     

    int RunCount=1;
    ros::init (argc, argv, "EVALUATION");
    ros::NodeHandle nh;
    PrettyPrint pp;
    ROS_INFO("Hello world: Simulated_User_General_Specific_Representation");

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
	|                                 |
	|     Randomly sort categories    |
	|_________________________________| */
	generateRrandomSequencesCategories(RunCount);
	

	//initialize the leveldb database
	_pdb = race_perception_db::PerceptionDB::getPerceptionDB(&nh); 
	string name = nh.getNamespace();
	ROS_INFO("PerceptionDB has been initialized ...");

	/* ________________________________________
	 |                                       |
	 |     read prameters from launch file   |
	 |_______________________________________| */
	// read database parameter
	nh.param<std::string>("/perception/home_address", home_address, "default_param");
	nh.param<int>("/perception/number_of_categories", number_of_categories, number_of_categories);
	nh.param<std::string>("/perception/name_of_approach", name_of_approach, "default_param");
	
	///read spin images parameters
	nh.param<int>("/perception/spin_image_width", spin_image_width, spin_image_width);
	nh.param<double>("/perception/spin_image_support_lenght", spin_image_support_lenght, spin_image_support_lenght);
	nh.param<int>("/perception/subsample_spinimages", subsample_spinimages, subsample_spinimages);
	nh.param<double>("/perception/keypoint_sampling_size", keypoint_sampling_size, keypoint_sampling_size);
	
	///read LDA parameters
	nh.param<double>("/perception/alpha", alpha, alpha);
	nh.param<double>("/perception/beta", beta, beta);
	nh.param<int>("/perception/total_number_of_topic", total_number_of_topic, total_number_of_topic);
	nh.param<int>("/perception/total_number_of_gibbs_sampling_iterations", total_number_of_gibbs_sampling_iterations, total_number_of_gibbs_sampling_iterations);
	nh.param<int>("/perception/general_dictionary_size", general_dictionary_size, general_dictionary_size);
	nh.param<int>("/perception/specific_dictionary_size", specific_dictionary_size, specific_dictionary_size);
    
	///read simulated teacher parameters
	nh.param<double>("/perception/P_Threshold", P_Threshold, P_Threshold);
	nh.param<int>("/perception/user_sees_no_improvment_const", user_sees_no_improvment_const, user_sees_no_improvment_const);
	nh.param<int>("/perception/window_size", window_size, window_size);

	///read the initial dictionary of visual words
	char buffer [500];
	int spin_image_support_lenght_float_tmp = spin_image_support_lenght *100;
	double m=sprintf (buffer, "%s%i%i%i%s","clusters", general_dictionary_size, spin_image_width, spin_image_support_lenght_float_tmp,".txt");
	string dictionary_name= buffer; 
	ROS_INFO ("dictionary_name = %s", dictionary_name.c_str());
	string IEETA_or_Washington = (home_address == "/home/hamidreza/") ? "IEETA/" : "Washington/";
	string initial_dictionary_path = ros::package::getPath("race_kfold_cross_validation_evaluation")+ "/generic_specific_dictionaries/" 
					+IEETA_or_Washington.c_str() + dictionary_name.c_str();
					
	vector <SITOV> initial_dictionary = readClusterCenterFromFile (initial_dictionary_path);
	ROS_INFO ("initial_dictionary_path.size() = %s", initial_dictionary_path.c_str());
	ROS_INFO ("initial_dictionary.size() = %ld", initial_dictionary.size());

	///define the path to specific dictionary
	string specific_dictionary_path = ros::package::getPath("race_kfold_cross_validation_evaluation") + "/generic_specific_dictionaries/"
					 + IEETA_or_Washington.c_str();
;
	
	
	
	unsigned int  track_id = 0;
// 	std::string categoryName;
	vector <int> total_number_of_words_in_each_object;
	vector <int> total_number_of_words_for_topic (total_number_of_topic); /// total number of words assigned to topic k 
	vector < vector <int> > sampled_topic_for_each_word_of_object ;
	vector < vector <int> > object_topic_matrix;
	vector < vector <int> > word_topic_matrix;
	vector < vector <double> > theta ;
	vector < vector <double> > phi;
	vector < double > posterior_distribution_topic (total_number_of_topic); /// 
	/* _____________________________________________________
	|                                                      |
	|  define train and test data using  cross validation  |
	|______________________________________________________| */

	///total number of object must be calculated atumaticly form train data.
	unsigned int total_number_of_train_object=0;
	
	
	/// we use 3/4 = 75% of data to train the LDA model
	int K_fold = 4;
	int iteration = 1;
	crossValidationData(K_fold, iteration, home_address);

	TrainDataForLDA( track_id, 
			  home_address, 
			  spin_image_width,
			  spin_image_support_lenght,
			  keypoint_sampling_size,
			  total_number_of_train_object, pp);
	      
	initialisation(  initial_dictionary, 
			 total_number_of_topic,
			 total_number_of_train_object,
			 total_number_of_words_in_each_object,
			 total_number_of_words_for_topic,
			 sampled_topic_for_each_word_of_object,
			 object_topic_matrix,
			 word_topic_matrix,
			 theta,
			 phi,
			 pp
		       );
			
	ROS_INFO ("LDA matrices have been initialized... ");


	
	estimateModel( total_number_of_gibbs_sampling_iterations,
			alpha /*constant*/,
			beta /*constant*/,
			initial_dictionary, 
			total_number_of_words_for_topic /*TODO: comput using object_topic_matrix*/,
			total_number_of_words_in_each_object /*TODO: comput using object_topic_matrix*/,
			sampled_topic_for_each_word_of_object,
			object_topic_matrix,    
			word_topic_matrix,
			posterior_distribution_topic,
			theta,
			phi,
			pp );
	
	
	ROS_INFO ("LDA Model has been initialized... ");
	ROS_INFO ("phi.size() = %ld , phi.at(0).size() = %ld", phi.size(),phi.at(0).size());

	
	/// Create a set of spin-images like topics using initial_dictionary dictionary and phi matrix
	createNewDictionaryFromPhi(initial_dictionary, phi, general_dictionary ); // general_dictionary = a set of spin-images like topics 
	ROS_INFO ("general_dictionary.size() = %i", general_dictionary.size());
	VisualizeNewDictionaryInMatlab(general_dictionary, spin_image_width, spin_image_support_lenght);
	
	/// new representation 
	deconceptualizingAllTrainData ();

	
	track_id =0;


	
	evaluationFile = ros::package::getPath("race_simulated_user")+ "/result/RUN"+ run_count + "/Detail_Evaluation.txt";
	Result.open (evaluationFile.c_str(), std::ofstream::out);
	//Result <<"\nclassification_threshold = " << classification_threshold <<  "\nspin_image_width = " << spin_image_width << "\nsubsample_spinimages = " << subsample_spinimages << "\n\n";
	
	string dataset= (home_address == "/home/hamidreza/") ? "IEETA" : "RGB-D Washington";
        Result  << "system configuration:"
		<< "\n\t-experiment_name = " << name_of_approach.c_str()
		<< "\n\t-name_of_dataset = " << dataset
		<< "\n\t-spin_image_width = "<<spin_image_width 
		<< "\n\t-spin_image_support_lenght = "<< spin_image_support_lenght
		<< "\n\t-keypoint_sampling_size = "<<keypoint_sampling_size
		<< "\n\t-number_of_topic = "<< total_number_of_topic
		<< "\n\t-specific_dictionary_size = "<< specific_dictionary_size
		<< "\n\t-alpha = "<< alpha
		<< "\n\t-beta = "<< beta
		<< "\n\t-total_number_of_gibbs_sampling_iterations = "<< total_number_of_gibbs_sampling_iterations
		<< "\n\t-recognition_threshold = "<< recognition_threshold
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

      /* _________________________________
	|                                |
	|       wait for 0.5 second      |
	|________________________________| */
	
	ros::Time start_time = ros::Time::now();
	while (ros::ok() && (ros::Time::now() - start_time).toSec() <1)
	{  //wait  
	}	
	
	/*________________________________
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

	/*_______________________________
	|                                |
	|        Introduce category      |
	|________________________________| */
	
	introduceNewCategoryUsingGeneralSpecificRepresentation(class_index,
							      track_id,
							      instance_number2.at(class_index-1),
							      evaluationFile,
							      pp,
							      spin_image_width,
							      spin_image_support_lenght,
							      (size_t) subsample_spinimages,
							      home_address,
							      general_dictionary,
							      specific_dictionary_size );
	
	number_of_instances+=3;
	number_of_taught_categories ++;
	category_introduced<< "1\n";

	/*_______________________________
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
	    //instance_number=1;
	    InstancePath= "";
	    if (introduceNewCategoryUsingGeneralSpecificRepresentation(class_index,
								      track_id,
								      instance_number2.at(class_index-1),
								      evaluationFile,
								      pp,
								      spin_image_width,
								      spin_image_support_lenght,
								      subsample_spinimages,							      
								      home_address,
								      general_dictionary,
								      specific_dictionary_size ) == -1)
	    {
		ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");
		ros::Duration duration = ros::Time::now() - beginProc;
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
	    bool User_sees_no_improvement_in_precision = false;	// In the current implementation, If the simulated teacher 
								// sees the precision doesn't improve in 100 iteration, then, 
								// it terminares evaluation of the system, originally, 
								// it was an empirical decision of the human instructor
									    
 	    
 	    

 	    
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
    
		// check the selected instance exist or not? if yes, send it to the race_object_representation
		if (InstancePath.size() < 2) 
		{
		    ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");
		    category_introduced.close();
		    ROS_INFO("\t\t[-]- Number of taught categories = %i", number_of_taught_categories); 		    
		    ROS_INFO ("Note: the experiment is terminated because there is not enough test data to continue the evaluation");

		    ros::Duration duration = ros::Time::now() - beginProc;
		    
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
			    ROS_ERROR("\t\t[-]- Could not read given object %s :",InstancePath.c_str());
			    return(0);
		    }
		    
		    ROS_INFO("\t\t[-]-  Track_id: %i , \tView_id: %i ",track_id, view_id );
		    
// 		    //Declare PCTOV msg 
// 		    boost::shared_ptr<race_perception_msgs::PCTOV> msg (new race_perception_msgs::PCTOV );
// 		    pcl::toROSMsg(*PCDFile, msg->point_cloud);
// 		    msg->track_id = track_id;//it is 
// 		    msg->view_id = view_id;
//    		    msg->ground_truth_name = InstancePath;
// 		    pub.publish (msg);
// 		    ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", InstancePath.c_str());
// 		    
// 		    
		    //Declare a boost share ptr to the spin image msg
		    boost::shared_ptr< vector <SITOV> > objectViewSpinImages;
		    objectViewSpinImages = (boost::shared_ptr< vector <SITOV> >) new (vector <SITOV>);
		    pp.info(std::ostringstream().flush() << "Given point cloud has " << PCDFile->points.size() << " points.");
	  
		    if (!estimateSpinImages( PCDFile, 
				0.01 /*downsampling_voxel_size*/, 
				0.05 /*normal_estimation_radius*/,
				spin_image_width /*spin_image_width*/,
				0.0 /*spin_image_cos_angle*/,
				1 /*spin_image_minimum_neighbor_density*/,
				spin_image_support_lenght /*spin_image_support_lenght*/,
				objectViewSpinImages,
				subsample_spinimages /*subsample spinimages*/
				))
		    {

			ROS_INFO( "Could not compute spin images");
			return 1;
		    }
		    

		    /// _________________________________________
		    ///                                         |
		    ///  Represention and recognition object   |
		    ///_______________________________________|
	     
		  
		    ///generic object representation
		    SITOV general_object_representation;
		    objectRepresentationBagOfWords (general_dictionary, *objectViewSpinImages, general_object_representation);

		    ///get list of all object categories
		    vector <ObjectCategory> ListOfObjectCategory = _pdb->getAllObjectCat();
		    vector <ObjectCategory> v_oc = _pdb->getAllObjectCat();
		    vector <NOCD> normalizedObjectCategoriesDistanceMsg;
		    vector <float> normalizedObjectCategoriesDistance;
		    pp.info(std::ostringstream().flush() << ListOfObjectCategory.size()<<" categories exist in the perception database" );

		    
		    for (size_t i = 0; i < ListOfObjectCategory.size(); i++) // all category exist in the database 
		    {
				if (ListOfObjectCategory.at(i).rtov_keys.size() > 1) 
				{    
					pp.info(std::ostringstream().flush() << ListOfObjectCategory.at(i).cat_name.c_str() <<" category has " 
									<< ListOfObjectCategory.at(i).rtov_keys.size()<< " views");
				
					///specific object representation		 
					vector <SITOV> specific_dictionary;
					string categoryName =  ListOfObjectCategory.at(i).cat_name.c_str();
					
					
					/// clusters7046_Bottle.txt
					char buffer [500];
					double n;
					int spin_image_support_lenght_float_tmp = spin_image_support_lenght *100;
					n=sprintf (buffer, "%s%i%i%i%s%s%s","clusters", specific_dictionary_size, spin_image_width, spin_image_support_lenght_float_tmp,"_",categoryName.c_str(),".txt");
					string dictionary_name= buffer; 
					ROS_INFO ("dictionary_name = %s", dictionary_name.c_str());
					string dictionary_path = specific_dictionary_path + dictionary_name;
					ROS_INFO ("dictionary_path = %s", dictionary_path.c_str());
					specific_dictionary = readClusterCenterFromFile (dictionary_path);    
					ROS_INFO ("specific_dictionary_size = %d", specific_dictionary.size());

					///specific object representation
					SITOV specific_object_representation;
					objectRepresentationBagOfWords (specific_dictionary, *objectViewSpinImages, specific_object_representation);
					ROS_INFO("\nsize of object view histogram %ld",specific_object_representation.spin_image.size());
							
					///concaticating generic and specific representation 
					SITOV object_representation_final;
					object_representation_final.spin_image.insert( object_representation_final.spin_image.end(), general_object_representation.spin_image.begin(), general_object_representation.spin_image.end());
					object_representation_final.spin_image.insert( object_representation_final.spin_image.end(), specific_object_representation.spin_image.begin(), specific_object_representation.spin_image.end());

							
					/// concatinating generic and specific dictionaries first and then create object representation
					// vector <SITOV> generic_specific_dictionary;
					// generic_specific_dictionary.insert( generic_specific_dictionary.end(), dictionary.begin(), dictionary.end());
					// generic_specific_dictionary.insert( generic_specific_dictionary.end(), specific_dictionary.begin(), specific_dictionary.end());		      
					// 
					// SITOV object_representation_final;
					// objectRepresentationBagOfWords (generic_specific_dictionary, *testObjectViewSpinImages, object_representation_final);
					// ROS_INFO("\nsize of object view histogram %ld",object_representation_final.spin_image.size());

					/// recognition    						
					///compute objectCategoryDistance() -> (return minimum distance and best_matched_index) 
					std::vector< SITOV > category_instances; 
					for (size_t j = 0; j < ListOfObjectCategory.at(i).rtov_keys.size(); j++)
					{
					vector< SITOV > objectViewHistogram = _pdb->getSITOVs(v_oc.at(i).rtov_keys.at(j).c_str());
					category_instances.push_back(objectViewHistogram.at(0));
					//pp.info(std::ostringstream().flush() << "size of object view histogram = " << objectViewHistogram.size());
					//pp.info(std::ostringstream().flush() << "key for the object view histogram = " << v_oc.at(i).rtov_keys.at(j).c_str());
					}
					pp.info(std::ostringstream().flush() << "number of object views histogram retrived from database = " << category_instances.size());
				
					///Compute the absolute distance from collected view and all category view instances
					float ObjectCategoryDistanec;
					int best_matched_index;
					float normalizedDistance;

					///euclidean distance
					//histogramBasedObjectCategoryDistance(object_representation_final,category_instances,ObjectCategoryDistanec,best_matched_index, pp);
					
					/// chi-squared distance
					chiSquaredBasedObjectCategoryDistance(object_representation_final,category_instances,ObjectCategoryDistanec,best_matched_index, pp);
					
					///KL distance
					//histogramBasedObjectCategoryKLDistance( object_representation_final,category_instances,ObjectCategoryDistanec,best_matched_index, pp);

					///Nearest neighbor classification
					normalizedObjectCategoryDistance(ObjectCategoryDistanec,1,normalizedDistance, pp);
								
					normalizedObjectCategoriesDistance.push_back(normalizedDistance);

					NOCD NOCDtmp;
					std::string oc_key = _pdb->makeOCKey(key::OC, ListOfObjectCategory.at(i).cat_name.c_str(), ListOfObjectCategory.at(i).cat_id);
					NOCDtmp.object_category_key= oc_key.c_str();
					NOCDtmp.normalized_distance=normalizedDistance;
					//NOCDtmp.best_matched_rtov_key= ListOfObjectCategory.at(i).rtov_keys.at(best_matched_index).c_str();
					NOCDtmp.cat_name = ListOfObjectCategory.at(i).cat_name;
					normalizedObjectCategoriesDistanceMsg.push_back(NOCDtmp);
				}
				else 
				{
					normalizedObjectCategoriesDistance.push_back(1000000);

					NOCD NOCDtmp;
					std::string oc_key = _pdb->makeOCKey(key::OC, ListOfObjectCategory.at(i).cat_name.c_str(), ListOfObjectCategory.at(i).cat_id);
					NOCDtmp.object_category_key= oc_key.c_str();
					NOCDtmp.normalized_distance=1000000;
					//NOCDtmp.best_matched_rtov_key= ListOfObjectCategory.at(i).rtov_keys.at(best_matched_index).c_str();
					NOCDtmp.cat_name = ListOfObjectCategory.at(i).cat_name;
					normalizedObjectCategoriesDistanceMsg.push_back(NOCDtmp);

					//pp.warn(std::ostringstream().flush() << "There is no data about object views in " <<ListOfObjectCategory.at(i).cat_name.c_str() << " category");
					pp.warn(std::ostringstream().flush() << "category size should larger than two, otherwise, we can not compute the ICD, ICD= 0.001");
				}
		    }

		    ///Print normalized distance from collected view to all category instances
		    for (size_t k = 0; k < normalizedObjectCategoriesDistanceMsg.size(); k++)
		    {
				pp.info(std::ostringstream().flush() << "NormalizedDistance (target_object, " 
					<< normalizedObjectCategoriesDistanceMsg.at(k).cat_name.c_str()<< " category) = "
					<< normalizedObjectCategoriesDistance.at(k));
		    }

		    //Timer toc
		    ros::Duration duration = ros::Time::now() - beginProc;
		    double duration_sec = duration.toSec();
		    pp.info(std::ostringstream().flush() << "Compute ND (O, Ci) took " << duration_sec << " secs");

	    
		    /// Classification Rule and Confidence value computation 
		    beginProc = ros::Time::now();
		    double sigma_distance = 0;
		    double recognition_threshold = 10000;
		    /// Sorting the normalizedObjectCategoriesDistanceMsg vector
		    for (size_t i = 0; i < normalizedObjectCategoriesDistanceMsg.size(); i++)
		    {
				for (size_t j = i; j < normalizedObjectCategoriesDistanceMsg.size(); j++)
				{	
					if (normalizedObjectCategoriesDistanceMsg.at(i).normalized_distance > normalizedObjectCategoriesDistanceMsg.at(j).normalized_distance )
					{
						NOCD NOCDtmp;
						NOCDtmp=normalizedObjectCategoriesDistanceMsg.at(i);
						normalizedObjectCategoriesDistanceMsg.at(i)=normalizedObjectCategoriesDistanceMsg.at(j);
						normalizedObjectCategoriesDistanceMsg.at(j)=NOCDtmp;
					}
				}
		    }


		    //confidenceValue = 1 - (minimumDistance/sigma_distance);
		    int categoryIndex = -1; //Added by mike (=-1) because of the if after and the function returns withount giving a value to categoryindex
		    float confidenceValue = 0;
		    float minimum_distance = 1000;
		    simpleClassificationRule(normalizedObjectCategoriesDistance, 
					categoryIndex, 
					confidenceValue, 
					(float)recognition_threshold,
					minimum_distance,
					pp);	

		    std::string result_string;
		    if (categoryIndex == -1)
		    {
			pp.info(std::ostringstream().flush() << "Predicted category is unknown" );
			result_string = "Unknown";
		    }
		    else
		    {
			pp.info(std::ostringstream().flush() << "Predicted category is "<<  ListOfObjectCategory.at(categoryIndex).cat_name.c_str());
			pp.info(std::ostringstream().flush() << "Confidence value = "<<  confidenceValue);
			result_string = ListOfObjectCategory.at(categoryIndex).cat_name.c_str();
		    }

		    RRTOV _rrtov;
		    _rrtov.header.stamp = ros::Time::now();
		    _rrtov.track_id = track_id;
		    _rrtov.view_id = view_id;
		    _rrtov.recognition_result = result_string;
		    _rrtov.minimum_distance = minimum_distance;
		    _rrtov.ground_truth_name = InstancePath.c_str();
		    _rrtov.result = normalizedObjectCategoriesDistanceMsg;

		    evaluationfunction(_rrtov);
		    
		    start_time = ros::Time::now();
		    while (ros::ok() && (ros::Time::now() - start_time).toSec() < 0.5)
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
// 	    	    
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
			
			monitor_precision (precision_file, F1);	
			
			if (F1 > P_Threshold)
			{
			    average_class_precision.push_back(F1);
			    User_sees_no_improvement_in_precision = false;
			    ROS_INFO("\t\t[-]- Precision= %f", Precision); 
//			    double Recall = TPtmp/double (TPtmp+FNtmp);
			    report_current_results(TPtmp,FPtmp,FNtmp,evaluationFile,false);
			    iterations = 1;
			    monitor_precision (local_F1_vs_learned_category, F1);
			    ros::spinOnce();		
			}  
			    
		    }//if
		    else if ( (iterations >= window_size * number_of_taught_categories)) // In this condition, if we are at iteration I>3n, we only
                                                                                         // compute precision as the average of last 3n, and discart the first
                                                                                         // I-3n iterations.
		    {
			//compute precision of last 3n, and discart the first I-3n iterations
			F1 = compute_precision_of_last_3n (recognition_results , number_of_taught_categories);
			ROS_INFO("\t\t[-]- Precision= %f", F1); 
		
			monitor_precision (precision_file, F1);						
			report_precision_of_last_3n (evaluationFile, F1);
			
			Result << "\n\t\t - F1 = "<< F1;

			if (F1 > P_Threshold)
			{
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
				average_class_precision.push_back(F1);
				User_sees_no_improvement_in_precision = true;
				ROS_INFO("\t\t[-]- User_sees_no_improvement_in_precision");
				ROS_INFO("\t\t[-]- Finish"); 
				ROS_INFO("\t\t[-]- Number of taught categories= %i", number_of_taught_categories); 
				
				Result.open (evaluationFile.c_str(), std::ofstream::app);
				Result << "\n After "<<user_sees_no_improvment_const<<" iterations, user sees no improvement in precision";
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
		      
			monitor_precision (precision_file, PrecisionSystem);
		    }
		    k++; // k<-k+1 : number of classification result
		    iterations	++;
		}//else	    	
	    }

	    monitor_F1_vs_learned_category (F1_vs_learned_category, TP, FP, FN );
	    ROS_INFO("\t\t[-]- Number of Iterations = %ld", iterations); 
	}
	ROS_INFO("\t\t[-]- Finish"); 
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



