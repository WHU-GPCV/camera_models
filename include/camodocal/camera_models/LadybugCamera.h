#ifndef LADYBUGCAMERA_H
#define LADYBUGCAMERA_H

#include <opencv2/core/core.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <vector>

#include <ceres/ceres.h>
#include "ceres/rotation.h"
using namespace ceres;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Camera.h"

// using namespace std;

namespace camodocal
{

#define LADYBUG_POLY_SIZE 12
#define LADYBUG_INV_POLY_SIZE 12

#define LADYBUG_CAMERA_NUM_PARAMS (LADYBUG_POLY_SIZE + LADYBUG_INV_POLY_SIZE + 3)

class LadybugCamera: public Camera
{
public:
    struct CostFunctor1 {
        CostFunctor1(double xx, double yy, double xxr, double yyr, double xx0rectified, double yy0rectified, double fInit)
            : x(xx), y(yy), xr(xxr), yr(yyr), x0rectified(xx0rectified), y0rectified(yy0rectified), 
            f(fInit), xequi(xr-x0rectified), yequi(yr-y0rectified) {}
        
        template<typename T>
        bool operator()(const T* const lamda, const T* const Ip0, const T* const k, const T* const p, const T* const c, T* residuals) const 
        { 
            T xt = T(x)-Ip0[0];
            T yt = T(y)-Ip0[1];
            T sqrtr=ceres::sqrt(xt*xt+yt*yt);
            T Rectifiedx=((T(f)*xt*ceres::tan(sqrtr/T(f)))/sqrtr);
            T Rectifiedy=((T(f)*yt*ceres::tan(sqrtr/T(f)))/sqrtr);
            T x_=lamda[0]*Rectifiedx;
            T y_=lamda[0]*Rectifiedy;
            T r2 = x_*x_ + y_*y_;
            T dx = x_*(k[0]*r2 + k[1]*r2*r2+k[2]*r2*r2*r2 + k[3]*r2*r2*r2*r2) + 2.0*p[0]*x_*y_+ p[1]*(r2+2.0*x_*x_)+ c[0]*x_ + c[1]*y_ ;
            T dy = y_*(k[0]*r2 + k[1]*r2*r2+k[2]*r2*r2*r2 + k[3]*r2*r2*r2*r2) + p[0]*(r2+2.0*y_*y_)+2.0*p[1]*x_*y_  +c[0]*y_ + c[1]*x_;
            //double xequi=xr-x0rectified;
            //double yequi=yr-y0rectified;

            residuals[0] = ceres::abs(x_-dx-T(xequi));
            residuals[1] = ceres::abs(y_-dy-T(yequi));

            return true;
        }
        
        private:
        // �۲�ֵ
        const double x;
        const double y;
        const double xr;
        const double yr;
        const double x0rectified;
        const double y0rectified;
        const double xequi;
        const double yequi;
        const double f;
    };

    struct CostFunctor2 {
        CostFunctor2(double xx, double yy, double xxr, double yyr, double xx0rectified, double yy0rectified, double fInit)
            : x(xx), y(yy), xr(xxr), yr(yyr), x0rectified(xx0rectified), y0rectified(yy0rectified), 
            f(fInit) {}
        
        template<typename T>
        bool operator()(const T* const lamda, const T* const Ip0, const T* const k, const T* const p, const T* const c, T* residuals) const 
        { 

            T x_ = T(xr)-T(x0rectified);
            T y_ = T(yr)-T(y0rectified);
            T r2 = x_*x_ + y_*y_;
            T dx = x_*(k[0]*r2 + k[1]*r2*r2+k[2]*r2*r2*r2 + k[3]*r2*r2*r2*r2) + 2.0*p[0]*x_*y_+ p[1]*(r2+2.0*x_*x_)+ c[0]*x_ + c[1]*y_ ;
            T dy = y_*(k[0]*r2 + k[1]*r2*r2+k[2]*r2*r2*r2 + k[3]*r2*r2*r2*r2) + p[0]*(r2+2.0*y_*y_)+2.0*p[1]*x_*y_  +c[0]*y_ + c[1]*x_;
            T xt=(x_+dx);
            T yt=(y_+dy);
            T sqrtr=ceres::sqrt(xt*xt+yt*yt);
            T Distortedx=((T(f)*xt*ceres::atan(sqrtr/(T(f))))/sqrtr);
            T Distortedy=((T(f)*yt*ceres::atan(sqrtr/(T(f))))/sqrtr);
            T xd=lamda[0]*Distortedx;
            T yd=lamda[0]*Distortedy;

            T xequi=T(x)-Ip0[0];
            T yequi=T(y)-Ip0[1];


            residuals[0]=ceres::abs(xd-xequi);
            residuals[1]=ceres::abs(yd-yequi);

            return true;
        }
        
        private:
        // �۲�ֵ
        const double x;
        const double y;
        const double xr;
        const double yr;
        const double x0rectified;
        const double y0rectified;
        const double f;
    };

    class Parameters: public Camera::Parameters
    {
    public:
        Parameters();

        double& poly(int idx) { return m_poly[idx]; }
        double& inv_poly(int idx) { return m_inv_poly[idx]; }

        double& frectified(void) { return m_frectified; }
        double& x0rectified(void) { return m_x0rectified; }
        double& y0rectified(void) { return m_y0rectified; }

        double frectified(void) const { return m_frectified; }
        double x0rectified(void) const { return m_x0rectified; }
        double y0rectified(void) const { return m_y0rectified; }

        double poly(int idx) const { return m_poly[idx]; }
        double inv_poly(int idx) const { return m_inv_poly[idx]; }

        bool readFromYamlFile(const std::string& filename);
        void writeToYamlFile(const std::string& filename) const;

        Parameters& operator=(const Parameters& other);
        friend std::ostream& operator<< (std::ostream& out, const Parameters& params);

    private:        
        double m_poly[LADYBUG_POLY_SIZE]; // 相机的畸变参数k1,k2,k3,k4,p1,p2,c1,c2, 缩放参数lamda, 鱼眼相机的焦距以及x0y0(fdistorted, x0distorted, y0distorted)
        double m_inv_poly[LADYBUG_INV_POLY_SIZE]; // 相机的畸变参数k1,k2,k3,k4,p1,p2,c1,c2, 缩放参数lamda, 鱼眼相机的焦距以及x0y0(fdistorted, x0distorted, y0distorted) 
        double m_frectified, m_x0rectified, m_y0rectified; // 纠正后图像的焦距和x0y0(frectified, x0rectified, y0rectified)

    };

    LadybugCamera();

    /**
    * \brief Constructor from the projection model parameters
    */
    LadybugCamera(const Parameters& params);

    Camera::ModelType modelType(void) const;
    const std::string& cameraName(void) const;
    int imageWidth(void) const;
    int imageHeight(void) const;

    void estimateIntrinsics(const std::vector<double> InnInitial,
                            const std::vector< std::vector<double> > Points);

    void estimateIntrinsics(const cv::Size& boardSize,
                            const std::vector< std::vector<cv::Point3f> >& objectPoints,
                            const std::vector< std::vector<cv::Point2f> >& imagePoints);

    // Lift points from the image plane to the sphere
    void liftSphere(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    //%output P

    // Lift points from the image plane to the projective space
    void liftProjective(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    //%output P

    // Projects 3D points to the image plane (Pi function)
    void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p) const;
    //%output p

    // Projects 3D points to the image plane (Pi function)
    // and calculates jacobian
    //void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
    //                  Eigen::Matrix<double,2,3>& J) const;
    //%output p
    //%output J

    void undistToPlane(const Eigen::Vector2d& p_u, Eigen::Vector2d& p) const;
    //%output p

    template <typename T>
    static void spaceToPlane(const T* const params,
                             const T* const q, const T* const t,
                             const Eigen::Matrix<T, 3, 1>& P,
                             Eigen::Matrix<T, 2, 1>& p);
    template <typename T>
    static void spaceToSphere(const T* const params,
                              const T* const q, const T* const t,
                              const Eigen::Matrix<T, 3, 1>& P,
                              Eigen::Matrix<T, 3, 1>& P_s);
    template <typename T>
    static void LiftToSphere(const T* const params,
                              const Eigen::Matrix<T, 2, 1>& p,
                              Eigen::Matrix<T, 3, 1>& P);

    template <typename T>
    static void SphereToPlane(const T* const params, const Eigen::Matrix<T, 3, 1>& P,
                               Eigen::Matrix<T, 2, 1>& p);


    void initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale = 1.0) const;
    cv::Mat initUndistortRectifyMap(cv::Mat& map1, cv::Mat& map2,
                                    float fx = -1.0f, float fy = -1.0f,
                                    cv::Size imageSize = cv::Size(0, 0),
                                    float cx = -1.0f, float cy = -1.0f,
                                    cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const;

    int parameterCount(void) const;

    const Parameters& getParameters(void) const;
    void setParameters(const Parameters& parameters);

    void readParameters(const std::vector<double>& parameterVec);
    void writeParameters(std::vector<double>& parameterVec) const;

    void writeParametersToYamlFile(const std::string& filename) const;

    std::string parametersToString(void) const;

    inline Eigen::Matrix<double, 1, 2> Rectifiedxy2Distortedxy(Eigen::Matrix<double, 1, 2> Rectifiedxy, double f);
    inline Eigen::Matrix<double, 1, 2> Distortedxy2Rectifiedxy(Eigen::Matrix<double, 1, 2> Distortedxy, double f);
    inline void EstimateF(std::vector<std::vector<double>> points, std::vector<double> InnInitial, int nWidth, int nHight, std::vector<std::vector<double>> &pointsForInitial, double &f, double &lamda);
    inline void EstimateFInv(std::vector<std::vector<double>> points, std::vector<double> InnInitial, int nWidth, int nHight, std::vector<std::vector<double>> &pointsForInitial, double &f, double &lamda);
    std::vector<std::vector<double>>  CalculateDistortPara(std::vector<std::vector<double>> points, std::vector<double> InnInitial, int nWidth_, int nHight_, std::vector<double> &Inn);
    void CalculateDistortParaInv(std::vector<std::vector<double>> points, std::vector<double> InnInitial, int nWidth_, int nHight_, std::vector<double> &InnInv, std::vector<std::vector<double>> pointsForInitial);
private:
    Parameters mParameters;
};

typedef boost::shared_ptr<LadybugCamera> LadybugCameraPtr;
typedef boost::shared_ptr<const LadybugCamera> LadybugCameraConstPtr;

// template <typename T>
// void
// LadybugCamera::spaceToPlane(const T* const params,
//                          const T* const q, const T* const t,
//                          const Eigen::Matrix<T, 3, 1>& P,
//                          Eigen::Matrix<T, 2, 1>& p)
// {
//     T P_c[3];
//     {
//         T P_w[3];
//         P_w[0] = T(P(0));
//         P_w[1] = T(P(1));
//         P_w[2] = T(P(2));

//         // Convert quaternion from Eigen convention (x, y, z, w)
//         // to Ceres convention (w, x, y, z)
//         T q_ceres[4] = {q[3], q[0], q[1], q[2]};

//         ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

//         P_c[0] += t[0];
//         P_c[1] += t[1];
//         P_c[2] += t[2];
//     }

//     T c = params[0];
//     T d = params[1];
//     T e = params[2];
//     T xc[2] = { params[3], params[4] };

//     //T poly[SCARAMUZZA_POLY_SIZE];
//     //for (int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
//     //    poly[i] = params[5+i];

//     T inv_poly[SCARAMUZZA_INV_POLY_SIZE];
//     for (int i=0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
//         inv_poly[i] = params[5 + SCARAMUZZA_POLY_SIZE + i];

//     T norm_sqr = P_c[0] * P_c[0] + P_c[1] * P_c[1];
//     T norm = T(0.0);
//     if (norm_sqr > T(0.0))
//         norm = std::sqrt(norm_sqr);

//     T theta = std::atan2(-P_c[2], norm);
//     T rho = T(0.0);
//     T theta_i = T(1.0);

//     for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
//     {
//         rho += theta_i * inv_poly[i];
//         theta_i *= theta;
//     }

//     T invNorm = T(1.0) / norm;
//     T xn[2] = {
//         P_c[0] * invNorm * rho,
//         P_c[1] * invNorm * rho
//     };

//     p(0) = xn[0] * c + xn[1] * d + xc[0];
//     p(1) = xn[0] * e + xn[1]     + xc[1];
// }

// template <typename T>
// void
// LadybugCamera::spaceToSphere(const T* const params,
//                           const T* const q, const T* const t,
//                           const Eigen::Matrix<T, 3, 1>& P,
//                           Eigen::Matrix<T, 3, 1>& P_s)
// {
//     T P_c[3];
//     {
//         T P_w[3];
//         P_w[0] = T(P(0));
//         P_w[1] = T(P(1));
//         P_w[2] = T(P(2));

//         // Convert quaternion from Eigen convention (x, y, z, w)
//         // to Ceres convention (w, x, y, z)
//         T q_ceres[4] = {q[3], q[0], q[1], q[2]};

//         ceres::QuaternionRotatePoint(q_ceres, P_w, P_c);

//         P_c[0] += t[0];
//         P_c[1] += t[1];
//         P_c[2] += t[2];
//     }

//     //T poly[SCARAMUZZA_POLY_SIZE];
//     //for (int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
//     //    poly[i] = params[5+i];

//     T norm_sqr = P_c[0] * P_c[0] + P_c[1] * P_c[1] + P_c[2] * P_c[2];
//     T norm = T(0.0);
//     if (norm_sqr > T(0.0))
//         norm = std::sqrt(norm_sqr);

//     P_s(0) = P_c[0] / norm;
//     P_s(1) = P_c[1] / norm;
//     P_s(2) = P_c[2] / norm;
// }

// template <typename T>
// void
// LadybugCamera::LiftToSphere(const T* const params,
//                           const Eigen::Matrix<T, 2, 1>& p,
//                           Eigen::Matrix<T, 3, 1>& P)
// {
//     T c = params[0];
//     T d = params[1];
//     T e = params[2];
//     T cc[2] = { params[3], params[4] };
//     T poly[SCARAMUZZA_POLY_SIZE];
//     for (int i=0; i < SCARAMUZZA_POLY_SIZE; i++)
//        poly[i] = params[5+i];

//     // Relative to Center
//     T p_2d[2];
//     p_2d[0] = T(p(0));
//     p_2d[1] = T(p(1));

//     T xc[2] = { p_2d[0] - cc[0], p_2d[1] - cc[1]};

//     T inv_scale = T(1.0) / (c - d * e);

//     // Affine Transformation
//     T xc_a[2];

//     xc_a[0] = inv_scale * (xc[0] - d * xc[1]);
//     xc_a[1] = inv_scale * (-e * xc[0] + c * xc[1]);

//     T norm_sqr = xc_a[0] * xc_a[0] + xc_a[1] * xc_a[1];
//     T phi = std::sqrt(norm_sqr);
//     T phi_i = T(1.0);
//     T z = T(0.0);

//     for (int i = 0; i < SCARAMUZZA_POLY_SIZE; i++)
//     {
//         if (i!=1) {
//             z += phi_i * poly[i];
//         }
//         phi_i *= phi;
//     }

//     T p_3d[3];
//     p_3d[0] = xc[0];
//     p_3d[1] = xc[1];
//     p_3d[2] = -z;

//     T p_3d_norm_sqr = p_3d[0] * p_3d[0] + p_3d[1] * p_3d[1] + p_3d[2] * p_3d[2];
//     T p_3d_norm = std::sqrt(p_3d_norm_sqr);

//     P << p_3d[0] / p_3d_norm, p_3d[1] / p_3d_norm, p_3d[2] / p_3d_norm;
// }

// template <typename T>
// void LadybugCamera::SphereToPlane(const T* const params, const Eigen::Matrix<T, 3, 1>& P,
//                                Eigen::Matrix<T, 2, 1>& p) {
//     T P_c[3];
//     {
//         P_c[0] = T(P(0));
//         P_c[1] = T(P(1));
//         P_c[2] = T(P(2));
//     }

//     T c = params[0];
//     T d = params[1];
//     T e = params[2];
//     T xc[2] = {params[3], params[4]};

//     T inv_poly[SCARAMUZZA_INV_POLY_SIZE];
//     for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++)
//         inv_poly[i] = params[5 + SCARAMUZZA_POLY_SIZE + i];

//     T norm_sqr = P_c[0] * P_c[0] + P_c[1] * P_c[1];
//     T norm = T(0.0);
//     if (norm_sqr > T(0.0)) norm = std::sqrt(norm_sqr);

//     T theta = std::atan2(-P_c[2], norm);
//     T rho = T(0.0);
//     T theta_i = T(1.0);

//     for (int i = 0; i < SCARAMUZZA_INV_POLY_SIZE; i++) {
//         rho += theta_i * inv_poly[i];
//         theta_i *= theta;
//     }

//     T invNorm = T(1.0) / norm;
//     T xn[2] = {P_c[0] * invNorm * rho, P_c[1] * invNorm * rho};

//     p(0) = xn[0] * c + xn[1] * d + xc[0];
//     p(1) = xn[0] * e + xn[1] + xc[1];
// }
}

#endif
