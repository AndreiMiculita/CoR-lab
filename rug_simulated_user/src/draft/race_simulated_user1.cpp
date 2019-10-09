/* _________________________________
   |                                 |
   |          RUN SYSTEM BY          |
   |_________________________________| */
   
//rm -rf /tmp/pdb
//roslaunch race_simulated_user race_simulated_user.launch


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
//#include <race_3d_object_tracking/TrackedObjectPointCloud.h>
#include <race_perception_msgs/perception_msgs.h>
#include <feature_extraction/spin_image.h>
//#include <race_3d_object_tracking/TrackedObjectPointCloud.h>
#include <object_conceptualizer/object_conceptualization.h>
#include <race_simulated_user/race_simulated_user_functionality.h>
#include <race_perception_utils/print.h>

/* _________________________________
  |                                 |
  |            NameSpace            |
  |_________________________________| */

using namespace pcl;
using namespace std;
using namespace ros;
using namespace race_perception_utils;

typedef pcl::PointXYZRGBA PointT;
#define Thershold 2

/* _________________________________
  |                                 |
  |         Global constant         |
  |_________________________________| */

#define spin_image_width 4
#define subsample_spinimages 10 /*subsample spinimages*/
#define classification_thershold 1

#define spin_image_support_lenght 0.2


/* _________________________________
  |                                 |
  |         Global Variable         |
  |_________________________________| */

PerceptionDB* _pdb;
typedef pcl::PointXYZRGBA PointT;

unsigned int cat_id = 1;
unsigned int track_id =1;
unsigned int view_id = 1;
string InstanceAddressTmp= "";

std::string PCDFileAddressTmp;
std::string True_Category_Global;
std::string Object_name_orginal;
std::string evaluationFile, evaluationTable, precision_file;

ofstream Result , Result_table , PrecisionMonitor;
int TP =0, FP=0, FN=0, TPtmp =0, FPtmp=0, FNtmp=0, Obj_Num=0 , track_id_gloabal = 1 , track_id_gloabal2=1;      

double PrecisionSystem;
    
void evaluationfunction(const race_perception_msgs::RRTOV &result)
{
    PrettyPrint pp;
    string True_cat = True_Category_Global;  
    
    std:: string Object_name;
    std:: string Temp_Object_name = Object_name_orginal;// for writing object name in result file;
    int rfind = Temp_Object_name.rfind("//")+2;       
    int len = Temp_Object_name.length();
    
    for (int i=0; i<(len-rfind); i++)
	Object_name += Temp_Object_name.at(i+rfind);
    
    if (len-rfind < 18)
    { 
	for (int i=len-rfind; i< 17;i++ )
	    Object_name += " ";
    }
    
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
    
    if (Predict_cat == "Unknown")
    {
	Predict_cat = "Category//Unk";
    }    
   
    pp.info(std::ostringstream().flush() << "[-]Object_name: "<< Object_name.c_str());
    pp.info(std::ostringstream().flush() << "[-]True_cat: "<<True_cat.c_str());
    pp.info(std::ostringstream().flush() << "[-]Predict_cat: " << Predict_cat.c_str());

    char Unknown[] = "Category//Unk";

    precision_file = ros::package::getPath("race_simulated_user")+ "/result/PrecisionMonitor.txt";
    PrecisionMonitor.open (precision_file.c_str(), std::ofstream::app);
    PrecisionMonitor.precision(4);
    

    Result.open( evaluationFile.c_str(), std::ofstream::app);    
    Result.precision(4);
    if ((strcmp(True_cat.c_str(),Unknown)!=0) && (strcmp(True_cat.c_str(),Predict_cat.c_str())==0))
    { 
	TP++;
	TPtmp++;
	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "1\t0\t0" << "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
	
	PrecisionSystem = TP/double (TP+FP);
	PrecisionMonitor << PrecisionSystem<<"\n";
    }
    else if ((strcmp(True_cat.c_str(),Unknown)!=0) && (strcmp(True_cat.c_str(),Predict_cat.c_str())!=0) && (strcmp(Predict_cat.c_str(),Unknown)!=0))
    {  	
	FP++; FN++;
	FPtmp++; FNtmp++;    
	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t1\t1"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
	Result << "\n\n\t.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
	Result << "\n\t - Correct classifier - Category Updated";
	Result << "\n\t .<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.\n\n";
	IntroduceNewCategory(InstanceAddressTmp, cat_id, track_id, view_id);
	track_id++;
	pp.info(std::ostringstream().flush() << "[-]Category Updated");
	
	PrecisionSystem = TP/double (TP+FP);
	PrecisionMonitor << PrecisionSystem<<"\n";

    }
    else if ((strcmp(True_cat.c_str(),Unknown)==0) && (strcmp(Predict_cat.c_str(),Unknown)!=0))
    { 	
	FP++; 
	FPtmp++;
	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t1\t0"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
	
	PrecisionSystem = TP/double (TP+FP);
	PrecisionMonitor << PrecisionSystem<<"\n";
    }
    else if ((strcmp(True_cat.c_str(),Unknown)!=0) && (strcmp(Predict_cat.c_str(),Unknown)==0))
    { 	
	FN++;
	FNtmp++;
	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t0\t1"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
	Result << "\n\n\t.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.";
	Result << "\n\t - Correct classifier - Category Updated";
	Result << "\n\t .<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.<>.\n\n";
	IntroduceNewCategory(InstanceAddressTmp, cat_id, track_id, view_id);
	track_id++;
	pp.info(std::ostringstream().flush() << "[-]Category Updated");
	
	PrecisionSystem = TP/double (TP+FP);
	PrecisionMonitor << PrecisionSystem<<"\n";
    }
    else if ((strcmp(True_cat.c_str(),Unknown)==0) && (strcmp(Predict_cat.c_str(),Unknown)==0))
    { 	
	Result << "\n"<<Obj_Num<<"\t"<<Object_name <<"\t\t"<< True_cat <<"\t\t"<< Predict_cat <<"\t\t"<< "0\t0\t0"<< "\t\t"<< minimumDistance;
	Result << "\n-----------------------------------------------------------------------------------------------------------------------------------";
	
	PrecisionSystem = TP/double (TP+FP);
	PrecisionMonitor << PrecisionSystem<<"\n";
	
    }
    Result.close();
    Result.clear();
    PrecisionMonitor.close();
    PrecisionMonitor.clear();
    pp.printCallback();
}


int main(int argc, char** argv)
{
           
    evaluationFile = ros::package::getPath("race_simulated_user")+ "/result/Detail_Evaluation.txt";
    Result.open (evaluationFile.c_str(), std::ofstream::app);
    Result <<"classification_thershold = " << classification_thershold <<  "\nspin_image_width = " << spin_image_width << "\nsubsample_spinimages = " << subsample_spinimages << "\n\n";
    Result << "Num"<<"\tObject_name" <<"\t\t\t"<< "True_Category" <<"\t\t"<< "Predict_Category"<< "\t"<< "TP" << "\t"<< "FP"<< "\t"<< "FN \t\tDistance";
    Result << "\n------------------------------------------------------------------------------------------------------------------------------------";
    Result.close();
    Result.clear();
    

    
    ros::init (argc, argv, "EVALUATION");

    ros::NodeHandle nh;
    _pdb = race_perception_db::PerceptionDB::getPerceptionDB(&nh); //initialize the database class_list_macros
                
    ros::Time start_time = ros::Time::now();
    while (ros::ok() && (ros::Time::now() - start_time).toSec() < 1)
    {  
	//wait
    }
		
    string name = nh.getNamespace();

    //create a subscriber to get recognition feedback
    unsigned found = name.find_last_of("/\\");
    std::string topic_name = name.substr(0,found) + "/tracking/recognition_result";
    ros::Subscriber sub = nh.subscribe(topic_name, 10, evaluationfunction);

    // create a publisher
    std::string pcin_topic = name.substr(0,found) + "/pipeline_default/tracker/tracked_object_point_cloud";  
    ros::Publisher pub = nh.advertise< race_perception_msgs::PCTOV> (pcin_topic, 1);

    /* _________________________________
    |                                 |
    |        Introduce category       |
    |_________________________________| */
    
    unsigned int c1_instance_number = 4;
    unsigned int c2_instance_number = 4;
    unsigned int c3_instance_number = 4;
    unsigned int c4_instance_number = 4;
    unsigned int c5_instance_number = 4;
    unsigned int c6_instance_number = 4;
    unsigned int c7_instance_number = 4;
    unsigned int c8_instance_number = 4;
    unsigned int c9_instance_number = 4;
    unsigned int c10_instance_number = 4;
    unsigned int c11_instance_number = 4;
    string categoryName="";
    
    string InstanceAddress= "";
    unsigned int class_index = 1; // class index
    unsigned int instance_number = 1;
    selectAnInstancefromSpecificCategory(class_index, 
					 instance_number, 
					 InstanceAddress);
    cat_id = 1;
    track_id =1;
    view_id = 1;
    ROS_INFO("\t\t[-]-Instance: %s", InstanceAddress.c_str());
    IntroduceNewCategory(InstanceAddress, cat_id, track_id, view_id); 
    
    int ffind = InstanceAddress.rfind("//")+2;       
    int lfind =  InstanceAddress.rfind("_");       
    for (int i=0; i<(lfind-ffind); i++)
    {
	categoryName += InstanceAddress.at(i+ffind);
    }
    Result.open (evaluationFile.c_str(), std::ofstream::app);
    Result << "\n\n\t################################################";
    Result << "\n\t\t -"<< categoryName.c_str() <<" Category Introduced";
    Result << "\n\t################################################\n\n";
    Result.close();
    Result.clear();
    
    track_id++;
//     view_id ++;
    instance_number=2;
    selectAnInstancefromSpecificCategory(class_index, 
					 instance_number, 
				         InstanceAddress);
    IntroduceNewCategory(InstanceAddress, cat_id, track_id, view_id);
    track_id++;
//     view_id ++;

    instance_number=3;
    selectAnInstancefromSpecificCategory(class_index, 
					 instance_number, 
				         InstanceAddress);
    IntroduceNewCategory(InstanceAddress, cat_id, track_id, view_id);
    track_id++;
//     view_id ++;
    
    int number_of_category = 1;
    float Precision_evaluation = 0;
    float P_Thereshold = 0.66;   
    while ( class_index < 10)
    {

	number_of_category ++;
	InstanceAddress= "";
	class_index ++; // class index
	instance_number=1;
// 	cat_id++;
	selectAnInstancefromSpecificCategory(class_index, 
					 instance_number, 
					 InstanceAddress);
	IntroduceNewCategory(InstanceAddress, cat_id, track_id, view_id);
    
	categoryName ="";
	int ffind = InstanceAddress.rfind("//")+2;       
	int lfind =  InstanceAddress.rfind("_");       
	for (int i=0; i<(lfind-ffind); i++)
	{
	    categoryName += InstanceAddress.at(i+ffind);
	}

	Result.open (evaluationFile.c_str(), std::ofstream::app);
	Result << "\n\n\t################################################";
	Result << "\n\t\t -"<< categoryName.c_str() <<" Category Introduced";
	Result << "\n\t################################################\n\n";
	Result.close();
	Result.clear();
	track_id++;
// 	view_id ++;
	instance_number=2;
	selectAnInstancefromSpecificCategory(class_index, 
					    instance_number, 
					    InstanceAddress);
	IntroduceNewCategory(InstanceAddress, cat_id, track_id, view_id);
	track_id++;
	instance_number=3;
	selectAnInstancefromSpecificCategory(class_index, 
					    instance_number, 
					    InstanceAddress);
	IntroduceNewCategory(InstanceAddress, cat_id, track_id, view_id);
	track_id++;
	
	TPtmp = 0;
	FPtmp = 0;
	FNtmp = 0;
	
	int k = 1; // number of classification result    
	float Precision_tmp = 1;
	Precision_evaluation = 0;
	unsigned int c =1;
	int itrations =0;

	while ( ((Precision_evaluation < P_Thereshold ) or (k < number_of_category)))
		//||((Precision_evaluation - Precision_tmp) < 0.05))
	{
	       
	    ROS_INFO("\t\t[-]+++++++++++++++++Iteration:%i",itrations+1);

	    InstanceAddress= "";
	    switch (c)
	    {
		case 1:
		    selectAnInstancefromSpecificCategory(c, 
						 c1_instance_number, 
						 InstanceAddress);
		    break;
		case 2 :
		    selectAnInstancefromSpecificCategory(c, 
						 c2_instance_number, 
						 InstanceAddress);
		    break;
		case 3:
		    selectAnInstancefromSpecificCategory(c, 
						 c3_instance_number, 
						 InstanceAddress);
		    break;
		case 4:
		    selectAnInstancefromSpecificCategory(c, 
						 c4_instance_number, 
						 InstanceAddress);
		    break;
		case 5:
		    selectAnInstancefromSpecificCategory(c, 
						 c5_instance_number, 
						 InstanceAddress);
		    break;
		case 6 :
		    selectAnInstancefromSpecificCategory(c, 
						 c6_instance_number, 
						 InstanceAddress);
		    break;
		case 7:
		    selectAnInstancefromSpecificCategory(c, 
						 c7_instance_number, 
						 InstanceAddress);
		    break;
		case 8:
		    selectAnInstancefromSpecificCategory(c, 
						 c8_instance_number, 
						 InstanceAddress);
		    break;
		case 9 :
		    selectAnInstancefromSpecificCategory(c, 
						 c9_instance_number, 
						 InstanceAddress);
		    break;
		case 10:
		    selectAnInstancefromSpecificCategory(c, 
						 c10_instance_number, 
						 InstanceAddress);
		    break;
	    }
	    
	    InstanceAddressTmp=InstanceAddress;
	    True_Category_Global = InstanceAddress;
	    True_Category_Global.resize(13);
	    
	    if (InstanceAddress.size() < 2)
	    {
		ROS_INFO("\t\t[-]-The file doesn't exist");
		ROS_INFO("\t\t[-]-Instance: %s", InstanceAddress.c_str());
		ROS_INFO("\t\t[-]-c: %i", c);
		ROS_INFO("\t\t[-]-class_index: %i", class_index);
		ROS_INFO("\t\t[-]-number_of_category: %i", number_of_category);
		c++;
	    }
	    else
	    {
		    InstanceAddress = ros::package::getPath("race_leave_one_out_evaluation") +"/"+ InstanceAddress.c_str();
		    ROS_INFO("\t\t[-]-Instance: %s", InstanceAddress.c_str());
		    
		    //load an instance from file
		    boost::shared_ptr<PointCloud<PointT> > PCDFile (new PointCloud<PointT>);
		    if (io::loadPCDFile <PointXYZRGBA> (InstanceAddress.c_str(), *PCDFile) == -1)
		    {	
			    ROS_ERROR("\t\t[-]-Could not read given object %s :",InstanceAddress.c_str());
			    return(0);
		    }
		    
		    ROS_INFO("\t\t[-]-  track_id: %i , \tview_id: %i ",track_id_gloabal2, view_id );

		    Object_name_orginal=InstanceAddress;    
		    //Declare PCTOV msg 
		    boost::shared_ptr<race_perception_msgs::PCTOV> msg (new race_perception_msgs::PCTOV );
		    pcl::toROSMsg(*PCDFile, msg->point_cloud);
		    msg->track_id = track_id_gloabal2;
		    msg->view_id = view_id;
		    pub.publish (msg);
		    ROS_INFO("\t\t[-]- Emulating race_object_tracking pakage by publish a point cloud: %s", 
			    InstanceAddress.c_str());
		    
		    // wait for half a second
		    start_time = ros::Time::now();
		    while (ros::ok() && (ros::Time::now() - start_time).toSec() < 0.5)
		    {  
			//wait
		    }
		    ros::spinOnce();
		    if (c == number_of_category)
		    {
			c = 1;
		    }
		    else
		    {
			c++;
		    }
		    
		    k++;
		    itrations++;
		    //ROS_INFO("\t\t[-]- Precision_evaluation= %f", Precision_evaluation); 
		    //Precision_evaluation = TPtmp/double (TPtmp+FPtmp);
		    Precision_evaluation =0;
		    
		    if ( (itrations >= 5*number_of_category) )
		    {
			Precision_evaluation = TPtmp/double (TPtmp+FPtmp);
			itrations = 0;
			Precision_tmp = Precision_evaluation;
			ROS_INFO("\t\t[-]- Precision_evaluation= %f", Precision_evaluation); 
			ROS_INFO("\t\t[-]- Precision_tmp= %f", Precision_tmp); 
	// 		double Precision = TPtmp/double (TPtmp+FPtmp);
	// 		double Recall = TPtmp/double (TPtmp+FNtmp);
			Result.open (evaluationFile.c_str(), std::ofstream::app);
			Result.precision(4);
			Result << "\n\n\t************************************************";
			Result << "\n\t\t - Sigma True  Positive = "<< TPtmp;
			Result << "\n\t\t - Sigma False Positive = "<< FPtmp;
			Result << "\n\t\t - Sigma False Negative = "<< FNtmp;
			Result << "\n\t\t - Precision  = "<< Precision_evaluation;//TPtmp/double (TPtmp+FPtmp);
	// 		Result << "\n\t\t - Recall = "<< Recall;//TPtmp/double (TPtmp+FNtmp);
			Result << "\n\n\t************************************************\n\n";
			Result << "\n------------------------------------------------------------------------------------------------------------------------------------";
			Result.close();
			ros::spinOnce();	
		    }
	    }	    
	    	    
	}
	
    }
    
    ROS_INFO("\t\t[-]- final"); 
    ROS_INFO("\t\t[-]- number of taught categories= %i", number_of_category); 

    
/*
 
 	ros::spinOnce();	
	double Precision = TPtmp/double (TPtmp+FPtmp);
	double Recall = TPtmp/double (TPtmp+FNtmp);
	Result.open (evaluationFile.c_str(), std::ofstream::app);
	Result.precision(4);
	Result << "\n\n\t************************************************";
	Result << "\n\t\t - Category Name : "<< categoryName.c_str();
	Result << "\n\t\t - Size of category : "<< category_size;
	Result << "\n\t\t - Sigma True  Positive = "<< TPtmp;
	Result << "\n\t\t - Sigma False Positive = "<< FPtmp;
	Result << "\n\t\t - Sigma False Negative = "<< FNtmp;
	Result << "\n\t\t - Precision  = "<< Precision;//TPtmp/double (TPtmp+FPtmp);
	Result << "\n\t\t - Recall = "<< Recall;//TPtmp/double (TPtmp+FNtmp);
	Result << "\n\n\t************************************************\n\n";
	Result << "\n------------------------------------------------------------------------------------------------------------------------------------";
	Result.close();

	if (categoryName!= "Category//Unk")
	{ 
	    double Category_ICD = Cat_ICD;
	    Result_table.open (evaluationTable.c_str(), std::ofstream::app);
	    Result_table.precision(4);
	    Result_table << categoryName.c_str()<<"\t     "<< category_size <<"\t\t" << Category_ICD <<"\t"<<Precision<<"\t\t"<< Recall;
	    Result_table << "\n----------------------------------------------------------------------\n";
	    Result_table.close();
	    Result_table.clear();
	}
	TPtmp=0;
	FPtmp=0;
	FNtmp=0;
	
    }//while1
    listOfObjectCategoriesAddress.close();
    listOfObjectCategoriesAddress.clear();
    
    ros::Duration(1).sleep(); // sleep for a second
    double GlobalPrecision = TP/double (TP+FP);
    double GlobalRecall = TP/double (TP+FN);
    ros::spinOnce();
    Result.open (evaluationFile.c_str(), std::ofstream::app);
    Result << "\n-------------------------------------------------------------------------------";
    Result << "\n\n\n===============================================================================";
    Result << "\n\t - Sigma True  Positive = "<< TP;
    Result << "\n\t - Sigma False Positive = "<< FP;
    Result << "\n\t - Sigma False Negative = "<< FN;
    Result << "\n\n\t - Precision  = "<< GlobalPrecision;
    Result << "\n\n\t - Recall = "<< GlobalRecall;
    Result << "\n\n===============================================================================";
    Result << "\n----------------------------------------------------------------------------------";
    Result << "\n----------------------------------------------------------------------------------";
    Result << "\n----------------------------------------------------------------------------------";
    Result << "\n----------------------------------------------------------------------------------";
    Result << "\n----------------------------------------------------------------------------------";
    Result << "\n----------------------------------------------------------------------------------";
    Result << "\n----------------------------------------------------------------------------------";
    Result << "\n----------------------------------------------------------------------------------";
    
    Result.close();
    
    Result_table.open (evaluationTable.c_str(), std::ofstream::app);
    Result_table.precision(4);
    Result_table << "Global       "<<"\t     "<< Obj_Num<<"\t\t" << "-" <<"\t\t"<<GlobalPrecision<<"\t\t"<< GlobalRecall;
    Result_table << "\n----------------------------------------------------------------------\n\n\n\n";
    Result_table.close();
    Result_table.clear();
    
*/
    
    
    return 1;
}
