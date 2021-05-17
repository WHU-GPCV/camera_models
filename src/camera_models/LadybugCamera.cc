#include "camodocal/camera_models/LadybugCamera.h"

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <iomanip>
#include <iostream>
// #include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "camodocal/gpl/gpl.h"



namespace camodocal
{

LadybugCamera::Parameters::Parameters()
 : Camera::Parameters(LADYBUG)
 , m_frectified(0.0)
 , m_x0rectified(0.0)
 , m_y0rectified(0.0)
{
    memset(m_poly, 0, sizeof(double) * LADYBUG_POLY_SIZE);
    memset(m_inv_poly, 0, sizeof(double) * LADYBUG_INV_POLY_SIZE);
}



bool
LadybugCamera::Parameters::readFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (!boost::iequals(sModelType, "ladybug"))
        {
            return false;
        }
    }

    m_modelType = LADYBUG;
    fs["camera_name"] >> m_cameraName;
    m_imageWidth = static_cast<int>(fs["image_width"]);
    m_imageHeight = static_cast<int>(fs["image_height"]);

    cv::FileNode n = fs["poly_parameters"];
    for(int i=0; i < LADYBUG_POLY_SIZE; i++)
        m_poly[i] = static_cast<double>(n[std::string("p") + boost::lexical_cast<std::string>(i)]);

    n = fs["inv_poly_parameters"];
    for(int i=0; i < LADYBUG_INV_POLY_SIZE; i++)
        m_inv_poly[i] = static_cast<double>(n[std::string("p") + boost::lexical_cast<std::string>(i)]);

    n = fs["projection_parameters"];
    m_frectified = static_cast<double>(n["frectified"]);
    m_x0rectified = static_cast<double>(n["x0rectified"]);
    m_y0rectified = static_cast<double>(n["y0rectified"]);

    return true;
}

void
LadybugCamera::Parameters::writeToYamlFile(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "model_type" << "ladybug";
    fs << "camera_name" << m_cameraName;
    fs << "image_width" << m_imageWidth;
    fs << "image_height" << m_imageHeight;

    fs << "poly_parameters";
    fs << "{";
    for(int i=0; i < LADYBUG_POLY_SIZE; i++)
        fs << std::string("p") + boost::lexical_cast<std::string>(i) << m_poly[i];
    fs << "}";

    fs << "inv_poly_parameters";
    fs << "{";
    for(int i=0; i < LADYBUG_INV_POLY_SIZE; i++)
        fs << std::string("p") + boost::lexical_cast<std::string>(i) << m_inv_poly[i];
    fs << "}";

    fs << "projection_parameters";
    fs << "{" << "frectified" << m_frectified
              << "x0rectified" << m_x0rectified
              << "y0rectified" << m_y0rectified << "}";

    fs.release();
}

LadybugCamera::Parameters&
LadybugCamera::Parameters::operator=(const LadybugCamera::Parameters& other)
{
    if (this != &other)
    {
        m_modelType = other.m_modelType;
        m_cameraName = other.m_cameraName;
        m_imageWidth = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;

        m_frectified = other.m_frectified;
        m_x0rectified = other.m_x0rectified;
        m_y0rectified = other.m_y0rectified;

        memcpy(m_poly, other.m_poly, sizeof(double) * LADYBUG_POLY_SIZE);
        memcpy(m_inv_poly, other.m_inv_poly, sizeof(double) * LADYBUG_INV_POLY_SIZE);
    }

    return *this;
}

std::ostream&
operator<< (std::ostream& out, const LadybugCamera::Parameters& params)
{
    out << "Camera Parameters:" << std::endl;
    out << "    model_type " << "ladybug" << std::endl;
    out << "   camera_name " << params.m_cameraName << std::endl;
    out << "   image_width " << params.m_imageWidth << std::endl;
    out << "  image_height " << params.m_imageHeight << std::endl;

    out << std::fixed << std::setprecision(10);

    out << "Poly Parameters" << std::endl;
    for(int i=0; i < LADYBUG_POLY_SIZE; i++)
        out << std::string("p") + boost::lexical_cast<std::string>(i) << ": " << params.m_poly[i] << std::endl;

    out << "Inverse Poly Parameters" << std::endl;
    for(int i=0; i < LADYBUG_INV_POLY_SIZE; i++)
        out << std::string("p") + boost::lexical_cast<std::string>(i) << ": " << params.m_inv_poly[i] << std::endl;

    out << "projection parameters" << std::endl;
    out << "            frectified " << params.m_frectified << std::endl
        << "            x0rectified " << params.m_x0rectified << std::endl
        << "            y0rectified " << params.m_y0rectified << std::endl;

    return out;
}

LadybugCamera::LadybugCamera()
{
}

LadybugCamera::LadybugCamera(const LadybugCamera::Parameters& params)
 : mParameters(params)
{
}

Camera::ModelType
LadybugCamera::modelType(void) const
{
    return mParameters.modelType();
}

const std::string&
LadybugCamera::cameraName(void) const
{
    return mParameters.cameraName();
}

int
LadybugCamera::imageWidth(void) const
{
    return mParameters.imageWidth();
}

int
LadybugCamera::imageHeight(void) const
{
    return mParameters.imageHeight();
}

void
LadybugCamera::estimateIntrinsics(const std::vector<double> InnInitial,
                               const std::vector< std::vector<double> > Points)
{
    Parameters params = getParameters();

    std::cout << "Inn ... ..." << std::endl;
    std::vector<double> Inn, InnInv;
    std::vector<std::vector<double>> pointsForInitial = CalculateDistortPara(Points, InnInitial, imageWidth(), imageHeight(), Inn);

    std::cout << "InvInn ... ..." << std::endl;
    CalculateDistortParaInv(Points, InnInitial, imageWidth(), imageHeight(), InnInv, pointsForInitial);

    for(int i=0; i<=LADYBUG_POLY_SIZE; ++i)
        params.poly(i) = Inn[i];
    for(int i=0; i<=LADYBUG_INV_POLY_SIZE; ++i)
        params.inv_poly(i) = InnInv[i];

    params.frectified() = Inn[LADYBUG_POLY_SIZE+0];
    params.x0rectified() = Inn[LADYBUG_POLY_SIZE+1];
    params.y0rectified() = Inn[LADYBUG_POLY_SIZE+2];

    setParameters(params);
    std::cout << "initial params:\n" << params << std::endl; 
}

void
LadybugCamera::estimateIntrinsics(const cv::Size& boardSize,
                               const std::vector< std::vector<cv::Point3f> >& objectPoints,
                               const std::vector< std::vector<cv::Point2f> >& imagePoints)
{
    
}

/** 
 * \brief Lifts a point from the image plane to the unit sphere
 *
 * \param p image coordinates
 * \param P coordinates of the point on the sphere
 */
void
LadybugCamera::liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{
    liftProjective(p, P);
    P.normalize();
}

/** 
 * \brief Lifts a point from the image plane to its projective ray
 *
 * \param p image coordinates
 * \param P coordinates of the projective ray
 */
void
LadybugCamera::liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const
{ 
    //畸变参数k1,k2,k3,k4,p1,p2,c1,c2
    double k1, k2, k3, k4, p1, p2, c1, c2;
    //尺度参数lamda
    double lamda;
    //鱼眼图像内定向参数(fdistorted, x0distorted, y0distorted)
    //矫正图像内定向参数(frectified，x0rectified, y0rectified)
    double fdistorted, frectified;
    double x0distorted, y0distorted;
    double x0rectified, y0rectified;

    //赋值得到每个内参参数
    k1 = mParameters.poly(0); k2 = mParameters.poly(1); k3 = mParameters.poly(2); k4 = mParameters.poly(3); p1 = mParameters.poly(4); p2 = mParameters.poly(5); c1 = mParameters.poly(6); c2 = mParameters.poly(7);
    lamda = mParameters.poly(8); fdistorted = mParameters.poly(9); x0distorted = mParameters.poly(10); y0distorted = mParameters.poly(11);
    frectified = mParameters.frectified(); x0rectified = mParameters.x0rectified(); y0rectified = mParameters.y0rectified();

    //开始纠正
    //首先转换为LadyBug坐标系(相当于图像坐标系逆时针旋转90度)下的平面坐标，此处使用的是鱼眼相机的x0,y0;
    double lx = p[0], ly = p[1];
    // ImgPixCoor2LadybugPixCoor(Pixalx, Pixaly, mImgWidth, mImgHeight, lx, ly);
    double xt = lx- x0distorted;
    double yt = ly - y0distorted;
    //使用等距投影模型进行投影将鱼眼图像投影到纠正后图像
    double sqrtr = std::sqrt(xt*xt + yt * yt);
    double Rectifiedx = ((fdistorted*xt*std::tan(sqrtr / fdistorted)) / sqrtr);
    double Rectifiedy = ((fdistorted*yt*std::tan(sqrtr / fdistorted)) / sqrtr);
    //按照缩放比例缩放至鱼眼相机同等大小的尺度上
    double x_ = lamda * Rectifiedx;
    double y_ = lamda * Rectifiedy;
    //使用图像畸变模型，计算畸变改正量，使用的畸变参数有k1,k2,k3,k4,p1,p2,c1,c2
    double r2 = x_ * x_ + y_ * y_;
    double dx = x_ * (k1*r2 + k2 * r2*r2 + k3 * r2*r2*r2 + k4 * r2*r2*r2*r2) + 2 * p1*x_*y_ + p2 * (r2 + 2 * x_*x_) + c1 * x_ + c2 * y_;
    double dy = y_ * (k1*r2 + k2 * r2*r2 + k3 * r2*r2*r2 + k4 * r2*r2*r2*r2) + p1 * (r2 + 2 * y_*y_) + 2 * p2*x_*y_ + c1 * y_ + c2 * x_;
    //进行畸变改正并转换为像素坐标,注意此处要使用纠正后影像的x0,y0;
    double xRectTemp= (x_ - dx) + x0rectified;
    double yRectTemp= (y_ - dy) + y0rectified;
    //从Ladybug的像素坐标转换为正常图像的像素坐标(相当于顺时针旋转90度)
    // LadybugPixCoor2ImgPixCoor(xRectTemp, yRectTemp, mImgWidth, mImgHeight, RectifiedPixalx, RectifiedPixaly);

    P << xRectTemp, yRectTemp, frectified;
}


/** 
 * \brief Project a 3D point (\a x,\a y,\a z) to the image plane in (\a u,\a v)
 *
 * \param P 3D point coordinates
 * \param p return value, contains the image point coordinates
 */
void
LadybugCamera::spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const
{
    //畸变参数k1,k2,k3,k4,p1,p2,c1,c2
    double k1, k2, k3, k4;
    double p1, p2;
    double c1, c2;
    //尺度参数lamda
    double lamda;
    //鱼眼图像内定向参数(fdistorted, x0distorted, y0distorted)
    //矫正图像内定向参数(frectified，x0rectified, y0rectified)
    double fdistorted, frectified;
    double x0distorted, y0distorted;
    double x0rectified, y0rectified;

    //赋值得到每个内参参数
    frectified = mParameters.frectified(); x0rectified = mParameters.x0rectified(); y0rectified = mParameters.y0rectified();
    k1 = mParameters.inv_poly(0); k2 = mParameters.inv_poly(1); k3 = mParameters.inv_poly(2); k4 = mParameters.inv_poly(3); p1 = mParameters.inv_poly(4); p2 = mParameters.inv_poly(5); c1 = mParameters.inv_poly(6); c2 = mParameters.inv_poly(7);
    lamda = mParameters.inv_poly(8); fdistorted = mParameters.inv_poly(9); x0distorted = mParameters.inv_poly(10); y0distorted = mParameters.inv_poly(11);

    double lx = P[0], ly = P[1];
    if(P[2] == frectified)
    {
        double scale = frectified/P[2];
        lx *= scale;
        ly *= scale;
    }
    
    // double LadybugPixalCoor_x, LadybugPixalCoor_y;
    // ImgPixCoor2LadybugPixCoor(Pixalx, Pixaly, mImgWidth, mImgHeight, LadybugPixalCoor_x, LadybugPixalCoor_y);
    double x_ = lx - x0rectified;
    double y_ = ly - y0rectified;
    double r2 = x_ * x_ + y_ * y_;
    double dx = x_ * (k1*r2 + k2 * r2*r2 + k3 * r2*r2*r2 + k4 * r2*r2*r2*r2) + 2 * p1*x_*y_ + p2 * (r2 + 2 * x_*x_) + c1 * x_ + c2 * y_;
    double dy = y_ * (k1*r2 + k2 * r2*r2 + k3 * r2*r2*r2 + k4 * r2*r2*r2*r2) + p1 * (r2 + 2 * y_*y_) + 2 * p2*x_*y_ + c1 * y_ + c2 * x_;
    double xt = x_ + dx;
    double yt = y_ + dy;
    double sqrtr = std::sqrt(xt*xt + yt * yt);
    double Distortedx = ((frectified*xt*std::atan(sqrtr / frectified)) / sqrtr);
    double Distortedy = ((frectified*yt*std::atan(sqrtr / frectified)) / sqrtr);
    double xd = lamda*Distortedx + x0distorted;
    double yd = lamda*Distortedy + y0distorted;

    p << xd, yd;
}


/** 
 * \brief Projects an undistorted 2D point p_u to the image plane
 *
 * \param p_u 2D point coordinates
 * \return image point coordinates
 */
void
LadybugCamera::undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const
{
    Eigen::Vector3d P(p_u[0], p_u[1], 1.0);
    spaceToPlane(P, p);
}


#if 0
void
LadybugCamera::initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale) const
{
    cv::Size imageSize(mParameters.imageWidth(), mParameters.imageHeight());

    cv::Mat mapX = cv::Mat::zeros(imageSize, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize, CV_32F);

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            double mx_u = m_inv_K11 / fScale * u + m_inv_K13 / fScale;
            double my_u = m_inv_K22 / fScale * v + m_inv_K23 / fScale;

            double xi = mParameters.xi();
            double d2 = mx_u * mx_u + my_u * my_u;

            Eigen::Vector3d P;
            P << mx_u, my_u, 1.0 - xi * (d2 + 1.0) / (xi + std::sqrt(1.0 + (1.0 - xi * xi) * d2));

            Eigen::Vector2d p;
            spaceToPlane(P, p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);
}
#endif

cv::Mat
LadybugCamera::initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                    float fx, float fy,
                                    cv::Size imageSize,
                                    float cx, float cy,
                                    cv::Mat rmat) const
{
    if (imageSize == cv::Size(0, 0))
    {
        imageSize = cv::Size(mParameters.imageWidth(), mParameters.imageHeight());
    }

    cv::Mat mapX = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);
    cv::Mat mapY = cv::Mat::zeros(imageSize.height, imageSize.width, CV_32F);

    Eigen::Matrix3f K_rect;

    K_rect << fx, 0, cx < 0 ? imageSize.width / 2 : cx,
              0, fy, cy < 0 ? imageSize.height / 2 : cy,
              0, 0, 1;

    if (fx < 0 || fy < 0)
    {
        throw std::string(std::string(__FUNCTION__) + ": Focal length must be specified");
    }

    Eigen::Matrix3f K_rect_inv = K_rect.inverse();

    Eigen::Matrix3f R, R_inv;
    cv::cv2eigen(rmat, R);
    R_inv = R.inverse();

    for (int v = 0; v < imageSize.height; ++v)
    {
        for (int u = 0; u < imageSize.width; ++u)
        {
            Eigen::Vector3f xo;
            xo << u, v, 1;

            Eigen::Vector3f uo = R_inv * K_rect_inv * xo;

            Eigen::Vector2d p;
            spaceToPlane(uo.cast<double>(), p);

            mapX.at<float>(v,u) = p(0);
            mapY.at<float>(v,u) = p(1);
        }
    }

    cv::convertMaps(mapX, mapY, map1, map2, CV_32FC1, false);

    cv::Mat K_rect_cv;
    cv::eigen2cv(K_rect, K_rect_cv);
    return K_rect_cv;
}

int
LadybugCamera::parameterCount(void) const
{
    return LADYBUG_CAMERA_NUM_PARAMS;
}

const LadybugCamera::Parameters&
LadybugCamera::getParameters(void) const
{
    return mParameters;
}

void
LadybugCamera::setParameters(const LadybugCamera::Parameters& parameters)
{
    mParameters = parameters;
}

void
LadybugCamera::readParameters(const std::vector<double>& parameterVec)
{
    if ((int)parameterVec.size() != parameterCount())
        return;

    Parameters params = getParameters();

    mParameters.frectified() = parameterVec.at(0);
    mParameters.x0rectified() = parameterVec.at(1);
    mParameters.y0rectified() = parameterVec.at(2);

    for (int i=0; i < LADYBUG_POLY_SIZE; i++)
        params.poly(i) = parameterVec.at(3+i);
    for (int i=0; i < LADYBUG_INV_POLY_SIZE; i++)
        params.inv_poly(i) = parameterVec.at(3 + LADYBUG_POLY_SIZE + i);

    setParameters(params);
}

void
LadybugCamera::writeParameters(std::vector<double>& parameterVec) const
{
    parameterVec.resize(parameterCount());
    parameterVec.at(0) = mParameters.frectified();
    parameterVec.at(1) = mParameters.x0rectified();
    parameterVec.at(2) = mParameters.y0rectified();
    for (int i=0; i < LADYBUG_POLY_SIZE; i++)
        parameterVec.at(3+i) = mParameters.poly(i);
    for (int i=0; i < LADYBUG_INV_POLY_SIZE; i++)
        parameterVec.at(3 + LADYBUG_POLY_SIZE + i) = mParameters.inv_poly(i);
}

void
LadybugCamera::writeParametersToYamlFile(const std::string& filename) const
{
    mParameters.writeToYamlFile(filename);
}

std::string
LadybugCamera::parametersToString(void) const
{
    std::ostringstream oss;
    oss << mParameters;

    return oss.str();
}

// Calib
std::vector<std::vector<double>>  
LadybugCamera::CalculateDistortPara(std::vector<std::vector<double>> points, std::vector<double> InnInitial, int nWidth_, int nHight_, std::vector<double> &Inn)
{
    int nWidth = nWidth_, nHight = nHight_; 
    double lamdaInitial, FocalLengthInitial;
    double frectified = InnInitial[4];
    double x0distorted=InnInitial[0];
    double y0distorted=InnInitial[1];
    double x0rectified=InnInitial[2];
    double y0rectified=InnInitial[3];
    std::vector<std::vector<double>> pointsForInitial;

    EstimateF(points,InnInitial,nWidth,nHight,pointsForInitial,FocalLengthInitial,lamdaInitial);

    Eigen::Matrix<double, Eigen::Dynamic, 8> Air;
    Air.resize(pointsForInitial.size()*2,8);
    Eigen::Matrix<double, Eigen::Dynamic, 1> Dir;
    Dir.resize(pointsForInitial.size()*2,1);
    //������������ʼֵ
    for(size_t i = 0; i<pointsForInitial.size(); i++)
    {
        double xi=pointsForInitial[i][0];
        double yi=pointsForInitial[i][1];
        double xri=pointsForInitial[i][2];
        double yri=pointsForInitial[i][3];
        double ri2=xi*xi+yi*yi;
        double ri4=ri2*ri2;
        double ri6=ri2*ri2*ri2;
        double ri8=ri2*ri2*ri2*ri2;
        Eigen::Matrix<double,1,8> Air1,Air2;
        Air1<<xi*ri2,xi*ri4,xi*ri6,xi*ri8,2*xi*yi,(ri2+2*xi*xi),xi,yi;
        Air2<<yi*ri2,yi*ri4,yi*ri6,yi*ri8,(ri2+2*yi*yi),2*xi*yi,yi,xi;
        Air.row(2*i)=Air1;
        Air.row(2*i+1)=Air2;
        Dir(2*i,0)=(xi-xri);
        Dir(2*i+1,0)=(yi-yri);
    }

    Eigen::Matrix<double, 8, 1> DistInitial=(Air.transpose()*Air).inverse()*(Air.transpose()*Dir);

    //����ʼֵ
    double lamda  = {lamdaInitial};
    double k[4] = {DistInitial(0), DistInitial(1), DistInitial(2), DistInitial(3)};
    double p[2] = {DistInitial(4), DistInitial(5)};
    double c[2] = {DistInitial(6), DistInitial(7)};
    double Ip0[2] = {x0rectified, y0rectified};

    Problem problem;
    for (size_t i = 0; i < points.size();++i)
    {
        problem.AddResidualBlock(
            new AutoDiffCostFunction<CostFunctor1,2,1,2,4,2,2>(
                new CostFunctor1(points[i][0],points[i][1], points[i][2],points[i][3], x0rectified, y0rectified, FocalLengthInitial)), NULL,
                &lamda,Ip0,k,p,c);
    }

    ceres::Solver::Options m_options;
    ceres::Solver::Summary m_summary;
    m_options.max_num_iterations = 1000;
    m_options.linear_solver_type = ceres::DENSE_QR;
    m_options.minimizer_progress_to_stdout = true;

    Solve(m_options, &problem,&m_summary);

    Inn.clear();
    Eigen::Matrix<double, 15, 1> result;
    result<<k[0],k[1],k[2],k[3],p[0],p[1],c[0],c[1],lamda,FocalLengthInitial,Ip0[0],Ip0[1],frectified,x0rectified,y0rectified;
    for(size_t i = 0;i<15;i++)
        Inn.push_back(result(i,0));
    
	return pointsForInitial;
}

void 
LadybugCamera::CalculateDistortParaInv(std::vector<std::vector<double>> points, std::vector<double> InnInitial, int nWidth_, int nHight_, std::vector<double> &InnInv, std::vector<std::vector<double>> pointsForInitial)
{
    int nWidth = nWidth_, nHight = nHight_; 
    double lamdaInitial, FocalLengthInitial;
    double frectified = InnInitial[4];
    double x0distorted=InnInitial[0];
    double y0distorted=InnInitial[1];
    double x0rectified=InnInitial[2];
    double y0rectified=InnInitial[3];
    std::vector<std::vector<double>> pointsForInitial1;

    EstimateFInv(points,InnInitial,nWidth,nHight,pointsForInitial1,FocalLengthInitial,lamdaInitial);

    Eigen::Matrix<double, Eigen::Dynamic, 8> Air;
    Air.resize(pointsForInitial.size()*2,8);
    Eigen::Matrix<double, Eigen::Dynamic, 1> Dir;
    Dir.resize(pointsForInitial.size()*2,1);
    //������������ʼֵ
    for(size_t i = 0; i<pointsForInitial.size(); i++)
    {
        double xi=pointsForInitial[i][0];
        double yi=pointsForInitial[i][1];
        double xri=pointsForInitial[i][2];
        double yri=pointsForInitial[i][3];
        double ri2=xri*xri+yri*yri;
        double ri4=ri2*ri2;
        double ri6=ri2*ri2*ri2;
        double ri8=ri2*ri2*ri2*ri2;
        Eigen::Matrix<double,1,8> Air1,Air2;
        Air1<<xri*ri2,xri*ri4,xri*ri6,xri*ri8,2*xri*yri,(ri2+2*xri*xri),xri,yri;
        Air2<<yri*ri2,yri*ri4,yri*ri6,yri*ri8,(ri2+2*yri*yri),2*xri*yri,yri,xri;
        Air.row(2*i)=Air1;
        Air.row(2*i+1)=Air2;
        Dir(2*i,0)=(xi-xri);
        Dir(2*i+1,0)=(yi-yri);
    }

    Eigen::Matrix<double, 8, 1> DistInitial=(Air.transpose()*Air).inverse()*(Air.transpose()*Dir);

    //����ʼֵ
    double lamda  = {lamdaInitial};
    double k[4] = {DistInitial(0), DistInitial(1), DistInitial(2), DistInitial(3)};
    double p[2] = {DistInitial(4), DistInitial(5)};
    double c[2] = {DistInitial(6), DistInitial(7)};
    double Ip0[2] = {x0rectified, y0rectified};

    Problem problem;
    for (size_t i = 0; i < points.size();++i)
    {
        problem.AddResidualBlock(
            new AutoDiffCostFunction<CostFunctor2,2,1,2,4,2,2>(
                new CostFunctor2(points[i][0],points[i][1], points[i][2],points[i][3], x0rectified, y0rectified, FocalLengthInitial)), NULL,
                &lamda,Ip0,k,p,c);
    }

    ceres::Solver::Options m_options;
    ceres::Solver::Summary m_summary;
    m_options.max_num_iterations = 1000;
    m_options.linear_solver_type = ceres::DENSE_QR;
    m_options.minimizer_progress_to_stdout = true;

    Solve(m_options, &problem,&m_summary);

    InnInv.clear();
    Eigen::Matrix<double, 15, 1> result;
    result<<k[0],k[1],k[2],k[3],p[0],p[1],c[0],c[1],lamda,FocalLengthInitial,Ip0[0],Ip0[1],frectified,x0rectified,y0rectified;
    for(size_t i = 0;i<15;i++){
        InnInv.push_back(result(i,0));
    }
}

inline Eigen::Matrix<double, 1, 2> 
LadybugCamera::Rectifiedxy2Distortedxy(Eigen::Matrix<double, 1, 2> Rectifiedxy, double f)
{
    double xr=Rectifiedxy(0,0);
    double yr=Rectifiedxy(0,1);
    double sqrtr=std::sqrt(xr*xr+yr*yr);
    double Distortedx=((f*xr*std::atan(sqrtr/f))/(sqrtr));
    double Distortedy=((f*yr*std::atan(sqrtr/f))/(sqrtr));
    Eigen::Matrix<double, 1, 2> Distortedxy;
    Distortedxy(0,0) = Distortedx;
    Distortedxy(0,1) = Distortedy;

    return Distortedxy;
}

inline Eigen::Matrix<double, 1, 2> 
LadybugCamera::Distortedxy2Rectifiedxy(Eigen::Matrix<double, 1, 2> Distortedxy, double f)
{
    double x_=Distortedxy(0,0);
    double y_=Distortedxy(0,1);
    double sqrtr=std::sqrt(x_*x_+y_*y_);
    double Rectifiedx=((f*x_*std::tan(sqrtr/f))/sqrtr);
    double Rectifiedy=((f*y_*std::tan(sqrtr/f))/sqrtr);
    Eigen::Matrix<double, 1, 2> Rectifiedxy;
    Rectifiedxy(0,0) = Rectifiedx;
    Rectifiedxy(0,1) = Rectifiedy;

    return Rectifiedxy;
}

inline void 
LadybugCamera::EstimateF(std::vector<std::vector<double>> points, std::vector<double> InnInitial, int nWidth, int nHight, std::vector<std::vector<double>> &pointsForInitial, double &f, double &lamda)
{
    double x0distorted=InnInitial[0];
    double y0distorted=InnInitial[1];
    double x0rectified=InnInitial[2];
    double y0rectified=InnInitial[3];
    Eigen::Matrix<double, 1, 2> xy0rectified;
    xy0rectified<<x0rectified,y0rectified;

    pointsForInitial.resize(points.size());

    double minsum;
    minsum=9999999;
    // ѡ�����ʵ�fֵ
    for(double ff = 910; ff<990; ff = ff+0.1)
    {
        Eigen::Matrix<double, 4, 2> DistortedConner;
        Eigen::Matrix<double, 4, 2> RectifiedConnerC;
        DistortedConner<<0,0,nWidth-1,0,0,nHight-1,nWidth-1,nHight-1;
        
        for(int i = 0; i<4; i++)
        {
            DistortedConner.row(i) = DistortedConner.row(i)-xy0rectified;
            RectifiedConnerC.row(i) = Distortedxy2Rectifiedxy(DistortedConner.row(i),ff);
        }
        
        double maxXDist=std::max(DistortedConner(0,0),std::max(DistortedConner(1,0),std::max(DistortedConner(2,0),DistortedConner(3,0))));
        double minXDist=std::min(DistortedConner(0,0),std::min(DistortedConner(1,0),std::min(DistortedConner(2,0),DistortedConner(3,0))));
        double maxYDist=std::max(DistortedConner(0,1),std::max(DistortedConner(1,1),std::max(DistortedConner(2,1),DistortedConner(3,1))));
        double minYDist=std::min(DistortedConner(0,1),std::min(DistortedConner(1,1),std::min(DistortedConner(2,1),DistortedConner(3,1))));

        double maxXRect=std::max(RectifiedConnerC(0,0),std::max(RectifiedConnerC(1,0),std::max(RectifiedConnerC(2,0),RectifiedConnerC(3,0))));
        double minXRect=std::min(RectifiedConnerC(0,0),std::min(RectifiedConnerC(1,0),std::min(RectifiedConnerC(2,0),RectifiedConnerC(3,0))));
        double maxYRect=std::max(RectifiedConnerC(0,1),std::max(RectifiedConnerC(1,1),std::max(RectifiedConnerC(2,1),RectifiedConnerC(3,1))));
        double minYRect=std::min(RectifiedConnerC(0,1),std::min(RectifiedConnerC(1,1),std::min(RectifiedConnerC(2,1),RectifiedConnerC(3,1))));
        int RectImgWidth=std::ceil(maxXRect-minXRect);
        int RectImgHeight=std::ceil(maxYRect-minYRect);
        double dispt=0;

		Eigen::Matrix<double, 1, 2> Distortedxy, RectifiedxyCal, Rectifiedxy;

        for(size_t i = 0;i<points.size();i++)
        {
            
            Distortedxy<<points[i][0]-x0rectified, points[i][1]-y0rectified;
            RectifiedxyCal=Distortedxy2Rectifiedxy(Distortedxy,ff);
            RectifiedxyCal=RectifiedxyCal*1.0*nWidth/RectImgWidth;
            Rectifiedxy<<points[i][2]-x0rectified,points[i][3]-y0rectified;
            dispt=dispt+(((Rectifiedxy-RectifiedxyCal).cwiseAbs()).sum());
        }
        
        if(dispt<minsum)
        {
            minsum=dispt;
            f=ff;
            lamda = 1.0*nWidth/RectImgWidth;
        }    
    }

    for(size_t i = 0;i<points.size();i++)
    {
        Eigen::Matrix<double, 1, 2> Distortedxy,RectifiedxyCal,Rectifiedxy;
        Distortedxy<<points[i][0]-x0rectified, points[i][1]-y0rectified;
        RectifiedxyCal=Distortedxy2Rectifiedxy(Distortedxy,f);
        RectifiedxyCal=RectifiedxyCal*lamda;
        Rectifiedxy<<points[i][2]-x0rectified,points[i][3]-y0rectified;
        std::vector<double> pfi_(4);
        pfi_[0] = RectifiedxyCal(0,0); pfi_[1] = RectifiedxyCal(0,1);
        pfi_[2] = Rectifiedxy(0,0); pfi_[3] = Rectifiedxy(0,1);
        pointsForInitial[i]=pfi_;
    }

}

inline void 
LadybugCamera::EstimateFInv(std::vector<std::vector<double>> points, std::vector<double> InnInitial, int nWidth, int nHight, std::vector<std::vector<double>> &pointsForInitial, double &f, double &lamda)
{
    double x0distorted=InnInitial[0];
    double y0distorted=InnInitial[1];
    double x0rectified=InnInitial[2];
    double y0rectified=InnInitial[3];
    Eigen::Matrix<double, 1, 2> xy0rectified;
    xy0rectified<<x0rectified,y0rectified;

    pointsForInitial.resize(points.size());

    //Eigen::Matrix<double, Eigen::Dynamic, 1> Distortedx=points.col(0);
    //Eigen::Matrix<double, Eigen::Dynamic, 1> Distortedy=points.col(1);
    //Eigen::Matrix<double, Eigen::Dynamic, 1> Rectifiedx=points.col(2);
    //Eigen::Matrix<double, Eigen::Dynamic, 1> Rectifiedy=points.col(3);

    double minsum;
    minsum=9999999;

    // ѡ�����ʵ�fֵ
    for(double ff = 390; ff<410; ff = ff+0.1)
    {
        Eigen::Matrix<double, 4, 2> RectifiedConner;
        Eigen::Matrix<double, 4, 2> DistortedConnerC;
        RectifiedConner<<0,0,nWidth-1,0,0,nHight-1,nWidth-1,nHight-1;
        
        for(int i = 0; i<4; i++)
        {
            RectifiedConner.row(i) = RectifiedConner.row(i)-xy0rectified;
            DistortedConnerC.row(i) = Rectifiedxy2Distortedxy(RectifiedConner.row(i),ff);
        }
        
        double maxXDist=std::max(DistortedConnerC(0,0),std::max(DistortedConnerC(1,0),std::max(DistortedConnerC(2,0),DistortedConnerC(3,0))));
        double minXDist=std::min(DistortedConnerC(0,0),std::min(DistortedConnerC(1,0),std::min(DistortedConnerC(2,0),DistortedConnerC(3,0))));
        double maxYDist=std::max(DistortedConnerC(0,1),std::max(DistortedConnerC(1,1),std::max(DistortedConnerC(2,1),DistortedConnerC(3,1))));
        double minYDist=std::min(DistortedConnerC(0,1),std::min(DistortedConnerC(1,1),std::min(DistortedConnerC(2,1),DistortedConnerC(3,1))));

        double maxXRect=std::max(RectifiedConner(0,0),std::max(RectifiedConner(1,0),std::max(RectifiedConner(2,0),RectifiedConner(3,0))));
        double minXRect=std::min(RectifiedConner(0,0),std::min(RectifiedConner(1,0),std::min(RectifiedConner(2,0),RectifiedConner(3,0))));
        double maxYRect=std::max(RectifiedConner(0,1),std::max(RectifiedConner(1,1),std::max(RectifiedConner(2,1),RectifiedConner(3,1))));
        double minYRect=std::min(RectifiedConner(0,1),std::min(RectifiedConner(1,1),std::min(RectifiedConner(2,1),RectifiedConner(3,1))));
        int DistortedImgWidth=std::ceil(maxXDist-minXDist);
        int DistortedImgHeight=std::ceil(maxYDist-minYDist);
        double dispt=0;
        for(size_t i = 0;i<points.size();i++)
        {
            Eigen::Matrix<double, 1, 2> Rectifiedxy,DistortedxyCal,Distortedxy;
            Rectifiedxy<<points[i][2]-x0rectified, points[i][3]-y0rectified;
            DistortedxyCal=Rectifiedxy2Distortedxy(Rectifiedxy,ff);
            DistortedxyCal=DistortedxyCal*1.0*nWidth/DistortedImgWidth;
            Distortedxy<< points[i][0]-x0rectified, points[i][1]-y0rectified;
            dispt=dispt+(((Distortedxy-DistortedxyCal).cwiseAbs()).sum());
        }
        
        if(dispt<minsum)
        {
            minsum=dispt;
            f=ff;
            lamda=1.0*nWidth/DistortedImgWidth;
        }    
    }

    for(size_t i = 0;i<points.size();i++)
    {
        Eigen::Matrix<double, 1, 2> Rectifiedxy,RectifiedxyCal,Distortedxy;
        Distortedxy<<points[i][0]-x0rectified, points[i][1]-y0rectified;
        RectifiedxyCal=Distortedxy2Rectifiedxy(Distortedxy,f);
        RectifiedxyCal=RectifiedxyCal*lamda;
        Rectifiedxy<<points[i][2]-x0rectified,points[i][3]-y0rectified;
        std::vector<double> pfi_(4);
        pfi_[0] = RectifiedxyCal(0,0); pfi_[1] = RectifiedxyCal(0,1);
        pfi_[2] = Rectifiedxy(0,0); pfi_[3] = Rectifiedxy(0,1);
        pointsForInitial[i]=pfi_;
    }
}

}
