#include<iostream>
#include<fstream>
#include<iomanip>
#include<dirent.h>
#include<string>

#include "camodocal/camera_models/LadybugCamera.h"

using namespace std;

template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;    
}

vector<string> getFiles(string cate_dir);

void LoadMap4Ladybug(const string &path, const int &nHight, const int &nWidth, cv::Mat &maps_x, cv::Mat &maps_y);
void LoadMap4LadybugBinary(const string &path, cv::Mat &maps_x, cv::Mat &maps_y);

void ImgPixCoor2LadybugPixCoor(double im_x, double im_y,  int ImgWidth, int ImgHeight, double &lad_x, double &lad_y){
    lad_x = im_y; lad_y = ImgWidth - im_x - 1;
}

void LadybugPixCoor2ImgPixCoor(double lad_x, double lad_y,  int ImgWidth, int ImgHeight, double &im_x, double &im_y){
    im_x = ImgHeight - lad_y - 1; im_y = lad_x;
}

int main(int argc, char **argv)
{
	if(argc < 5)
	{
		cout << "./LBCalib   Dir_path_D2U   Dir_path_U2D InitParampath scale  tranlate (eg. ./LBCalib ../testdata/D2U_ALL_1616X1232/ ../testdata/U2D_ALL_1616X1232/ ../testdata/InitParam.txt 4 1 )" << endl;
		exit(0);
	}

	vector<string> strPrefix_d2u;
    strPrefix_d2u=getFiles(string(argv[1]));
	const int nCams = (int)strPrefix_d2u.size();

	vector<string> strPrefix_u2d;
    strPrefix_u2d=getFiles(string(argv[2]));

	double scale = stringToNum<double>(argv[4]);
	std::cout << "isTranlate:"<<argv[5] << std::endl;
	bool isTranlate = (argv[5] == "1");

	if((int)strPrefix_u2d.size() != nCams)
	{
		cout << " the number of d2u is  not equal with the number of u2d!!!  " << endl;
		exit(0);
	}

    // string Innoutpath = "./InnPara.txt";
    // string InvInnoutpath = "./InvInnPara.txt";
	string InitParampath = string(argv[3]);

	ifstream in_initpara(InitParampath);
	if(!in_initpara.is_open())
	{
		cout << "InitParampath(" << InitParampath << ") has no file!!! " << endl;
		exit(0);
	}

	std::vector<double> x0rectifieds;
	std::vector<double> y0rectifieds;
	std::vector<double> frectifieds;
	while(!in_initpara.eof())
	{
		string s;
		getline(in_initpara, s);

		if(s == "" || s[0] == '#')
			continue;

		stringstream ss;
		ss << s;

		int index; 
		double x0rectified, y0rectified, frectified;

		ss >> index; 
		ss >> x0rectified; ss >> y0rectified; ss >> frectified;

		x0rectifieds.push_back(x0rectified);
		y0rectifieds.push_back(y0rectified);
		frectifieds.push_back(frectified);
	}

	// optimizer
	cout << endl << "===========start optimization of Inn-InvInn Parameters =====================" << endl;
	google::InitGoogleLogging(argv[0]);

	std::vector<std::vector<double>> Inn(nCams);
	std::vector<std::vector<double>> InnInv(nCams);
	// std::ofstream out1(Innoutpath);
	// std::ofstream out2(InvInnoutpath);
	for (int i = 0; i < nCams; i++)
	{
		int nWidth, nHight;
		nWidth = 1616;
		nHight = 1232;

		// For D2U
		string path_D2U = string(argv[1]) + "/" + strPrefix_d2u[i];
		cv::Mat map_x_d2u, map_y_d2u;
		LoadMap4Ladybug(path_D2U, nHight, nWidth, map_x_d2u, map_y_d2u);

		
		// std::ifstream in1(path_D2U);
		// if(!in1.is_open())
		// {
		// 	cout << "Can't open" << path_D2U << endl;
		// 	exit(0);
		// }

		// cout << "Reading " << path_D2U <<endl;

		// string s; 
		// getline(in1, s);

		// std::vector<std::vector<std::pair<double, double>>> maps_d2u;
		// for(int y = 0; y < nHight; y++)
		// {
		// 	std::vector<std::pair<double, double>> t;
		// 	for(int x = 0; x< nWidth; x++)
		// 	{
		// 		getline(in1, s);
		// 		stringstream ss;
		// 		ss << s;
		// 		int v1, v2; double v3, v4;
		// 		ss >> v1; ss >> v2;
		// 		std::cout << s << std::endl;
		// 		std::cout << v1 << " " << v2 << " " << x << " " << y << std::endl;
		// 		if(v1 != y || v2 != x){
					
		// 			cout<<"LLLLLLLLLLLLLLLLLLL"<<endl;
		// 			exit(0);
		// 		}

		// 		ss >> v3; ss >> v4;
		// 		std::pair<double, double> tt(v4, v3);
		// 		t.push_back(tt);
		// 	}
		// 	maps_d2u.push_back(t);
		// }

		// in1.close();

		// For U2D
		string path_U2D = string(argv[2]) + "/" + strPrefix_u2d[i];
		cv::Mat map_x_u2d, map_y_u2d;
		LoadMap4Ladybug(path_U2D, nHight, nWidth, map_x_u2d, map_y_u2d);


		// std::ifstream in2(path_U2D);
		// if(!in2.is_open())
		// {
		// 	cout << "Can't open" << path_U2D << endl;
		// 	exit(0);
		// }

		// cout << "Reading " << path_U2D <<endl;

		// getline(in2, s);

		// std::vector<std::vector<std::pair<double, double>>> maps_u2d;
		// for(int y = 0; y < nHight; y++)
		// {
		// 	std::vector<std::pair<double, double>> t;
		// 	for(int x = 0; x< nWidth; x++)
		// 	{
		// 		getline(in2, s);
		// 		stringstream ss;
		// 		ss << s;
		// 		int v1, v2; double v3, v4;
		// 		ss >> v1; ss >> v2;
		// 		ss >> v3; ss >> v4;
		// 		if(v1 != y || v2 != x){
		// 			cout<<"LLLLLLLLLLLLLLLLLLL"<<endl;
		// 			exit(0);
		// 		}

		// 		std::pair<double, double> tt(v4, v3);
		// 		t.push_back(tt);
		// 	}
		// 	maps_u2d.push_back(t);
		// }

		// in2.close();

		// Input InitParam
		double x0distorted, y0distorted, x0rectified, y0rectified, frectified;

		/*
		cout << "Please input x0rectified y0rectified frectified:" <<endl;
		cin >> x0rectified >> y0rectified >> frectified;
		cout << endl;*/

		x0rectified = x0rectifieds[i]; y0rectified = y0rectifieds[i]; frectified = frectifieds[i];
		int x0r_i = (int)round(x0rectified); int y0r_i = (int)round(y0rectified); 
		// std::pair<double, double> ii =  maps_u2d[y0r_i][x0r_i];
		std::pair<double, double> ii(map_x_u2d.at<double>(y0r_i, x0r_i), map_y_u2d.at<double>(y0r_i, x0r_i));
		x0distorted = ii.first; y0distorted = ii.second;

		

		vector<double> InnInitial;
		InnInitial.resize(6);
		InnInitial[0] = x0distorted / scale;
		InnInitial[1] = y0distorted / scale;
		InnInitial[2] = x0rectified / scale;
		InnInitial[3] = y0rectified / scale;
		InnInitial[4] = frectified / scale;

		if(isTranlate)
		{
			double x0distorted_, y0distorted_, x0rectified_, y0rectified_;
			LadybugPixCoor2ImgPixCoor(x0distorted, y0distorted, nWidth, nHight, x0distorted_, y0distorted_);
			LadybugPixCoor2ImgPixCoor(x0rectified, y0rectified, nWidth, nHight, x0rectified_, y0rectified_);
			InnInitial[0] = x0distorted_ / scale;
			InnInitial[1] = y0distorted_ / scale;
			InnInitial[2] = x0rectified_ / scale;
			InnInitial[3] = y0rectified_ / scale;
		}

		std::cout << InnInitial[0] << ", "<< InnInitial[1] << ", "<< InnInitial[2] << ", "<< InnInitial[3] << ", "<< InnInitial[4] << std::endl;

		// Start  Optmize
		cout << "optmize " << i + 1 << " camera!" << endl;
		std::vector<std::vector<double>> points;
		for (int x = 57; x < nWidth; x = x + 150)
		{
			for (int y = 75; y < nHight; y = y + 120)
			{
				double Rectifiedx, Rectifiedy;
				// std::pair<double, double> ii =  maps_d2u[y][x];
				std::pair<double, double> ii(map_x_d2u.at<double>(y,x), map_y_d2u.at<double>(y,x));
				Rectifiedx = ii.first;
				Rectifiedy = ii.second;

				double n_x, n_y, n_r_x, n_r_y;
				if(isTranlate)
				{
					LadybugPixCoor2ImgPixCoor(x, y, nWidth, nHight, n_x, n_y);
					LadybugPixCoor2ImgPixCoor(Rectifiedx, Rectifiedy, nWidth, nHight, n_r_x, n_r_y);
				}
				else
				{
					n_x = x;
					n_y = y;
					n_r_x = Rectifiedx;
					n_r_y = Rectifiedy;
				}

				std::vector<double> drp;
				drp.resize(4);
				drp[0] = n_x / scale;
				drp[1] = n_y / scale;
				drp[2] = n_r_x / scale;
				drp[3] = n_r_y / scale;
				
				points.emplace_back(drp);
			}
		}

		camodocal::LadybugCameraPtr camera(new camodocal::LadybugCamera);
		camodocal::LadybugCamera::Parameters params = camera->getParameters( );
		params.cameraName( )                 = "Cam" + std::to_string(i);
		params.imageWidth( )                 = isTranlate? nHight/scale:nWidth / scale;
		params.imageHeight( )                = isTranlate? nWidth/scale:nHight / scale;
		camera->setParameters( params );
		camera->estimateIntrinsics(InnInitial, points);

		std::string save_name;
		if(isTranlate)
		{
			stringstream ss; ss << std::fixed << std::setprecision(1) << "ladybug3_cam"+std::to_string(i)+"_" << scale << "_t.yaml";
			save_name = ss.str();
		}
		else
		{
			stringstream ss; ss << std::fixed << std::setprecision(1) << "ladybug3_cam"+std::to_string(i)+"_" << scale << ".yaml";
			save_name = ss.str();
		}

		camera->writeParametersToYamlFile(save_name);

		// cout << "Inn ... ..." << endl;
		// std::vector<std::vector<double>> pointsForInitial = CalculateDistortPara(points, InnInitial, nWidth, nHight, Inn[i]);
		// cout << "InvInn ... ..." << endl;
		// CalculateDistortParaInv(points, InnInitial, nWidth, nHight, InnInv[i], pointsForInitial);

		// out1 << i << " ";
		// for (size_t n = 0; n < Inn[i].size(); n++) {

		// 	if (n != Inn[i].size() - 1) {
		// 		out1 << Inn[i][n] << " ";
		// 	}
		// 	else {
		// 		out1 << Inn[i][n] << std::endl;
		// 	}
		// }
		// out1 << endl;

		// out2 << i << " ";
		// for (size_t n = 0; n < InnInv[i].size(); n++) {

		// 	if (n != InnInv[i].size() - 1) {
		// 		out2 << InnInv[i][n] << " ";
		// 	}
		// 	else {
		// 		out2 << InnInv[i][n] << std::endl;
		// 	}
		// }
		// out2 << endl;
	}

	// out1.close();
	// out2.close();
	cout <<endl << "============================================="<< endl 
		<< "Inn , InvInn and Ex parameters output!" << endl << endl;

    return 0;
}


vector<string> getFiles(string cate_dir)
{
    vector<string> files;//存放文件名

    DIR *dir;
    struct dirent *ptr;

    if ((dir=opendir(cate_dir.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr=readdir(dir)) != NULL)
    {
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
                continue;
        files.push_back(ptr->d_name);
    }
    closedir(dir);

    //排序，按从小到大排序
    sort(files.begin(), files.end());
    return files;
}


void LoadMap4Ladybug(const string &path, const int &nHight, const int &nWidth, cv::Mat &maps_x, cv::Mat &maps_y)
{
    std::ifstream in1(path);
    // std::ofstream out1(path.substr( 0, path.find_last_of('.') ) + string("_") + to_string(i++) + string(".bin"), std::ios::binary);
    // std::ofstream out(path.substr( 0, path.find_last_of('.') ) + to_string(i++) + string(".txt"));
 
    double* data = new double[2+nWidth*nHight*4];
    int w = nWidth; int h = nHight;
    double wf = w; double hf = h;
    data[0] = hf; data[1] = wf;
    std::cout << data[0] << "," << data[1] << std::endl;

    if(!in1.is_open())
    {
        cout << "Can't open " << path << endl;
        exit(0);
    }

    cout << "Reading " << path << endl;

    string s; 
    getline(in1, s);
    // out << s << endl;

    maps_x.release(); maps_x.release();
    maps_x.create(nHight, nWidth, CV_64FC1);     
    maps_y.create(nHight, nWidth, CV_64FC1);
    for(int y = 0; y < nHight; y++)
    {
        for(int x = 0; x < nWidth; x++)
        {
            getline(in1, s);
            // out << s << endl;
            stringstream ss;
            ss << s;
            int v1, v2; double v3, v4;
            ss >> v1; ss >> v2;
            if(v1 != y || v2 != x){
                cout<<"Wrong in the LoadMap4Ladybug func..."<<endl;
                exit(0);
            }

            ss >> v3; ss >> v4;
            maps_x.at<double>(y, x) = v4;
            maps_y.at<double>(y, x) = v3;

            data[2+(y*nWidth+x)*4+0] = (double)v1; data[2+(y*nWidth+x)*4+1] = (double)v2;
            data[2+(y*nWidth+x)*4+2] = v3; data[2+(y*nWidth+x)*4+3] = v4;
            // std::cout << data[2+(y*nWidth+x)*4+0] << "," << data[2+(y*nWidth+x)*4+1] << "," << data[2+(y*nWidth+x)*4+2] << "," << data[2+(y*nWidth+x)*4+3] << std::endl;
        }
    }

    in1.close();
    // out.close();

    // std::cout << sizeof(data) << std::endl;
    // out1.write(reinterpret_cast<char*>(data), (2+nWidth*nHight*4)*sizeof(double));
    // out1.close();
}


void LoadMap4LadybugBinary(const string &path, cv::Mat &maps_x, cv::Mat &maps_y)
{
    std::ifstream in1(path, std::ios::binary);
    if(!in1.is_open())
    {
        cout << "Can't open " << path << endl;
        exit(0);
    }
    cout << "Reading " << path << endl;
    double size[2]; in1.read(reinterpret_cast<char*>(size), sizeof(double) * 2);
    int h = static_cast<int>(size[0]), w = static_cast<int>(size[1]);
    double* data = new double[4*h*w]; in1.read(reinterpret_cast<char*>(data), sizeof(double) * (4*h*w));
    in1.close();

    maps_x.release(); maps_x.release();
    maps_x.create(h, w, CV_64FC1);     
    maps_y.create(h, w, CV_64FC1);
    for(int y = 0; y < h; y++)
    {
        for(int x = 0; x < w; x++)
        {
            int v1, v2; double v3, v4;
            v1 = (int)data[(y*w+x)*4+0]; v2 = (int)data[(y*w+x)*4+1];
            if(v1 != y || v2 != x){
                cout<<"Wrong in the LoadMap4Ladybug func..."<<endl;
                exit(0);
            }

            v3 = data[(y*w+x)*4+2]; v4 = data[(y*w+x)*4+3];
            maps_x.at<double>(y, x) = v4;
            maps_y.at<double>(y, x) = v3;
        }
    }
}