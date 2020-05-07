
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <set>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/segmentation/extract_clusters.h>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (const cv::DMatch& kptMatch: kptMatches)
    {
        int currIndex = kptMatch.trainIdx, prevIndex = kptMatch.queryIdx;
        cv::KeyPoint kptCurr = kptsCurr[currIndex], kptPrev = kptsPrev[prevIndex];
        if (boundingBox.roi.contains(kptCurr.pt))
        {
            boundingBox.kptMatches.push_back(kptMatch);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // calculate distance ratios between all permutations of key point matches
    vector<double> distanceRatios;
    const double minDistance = 100.0;
    cout << "   camera: processing " << kptMatches.size() << " keypoint matches";
    for (auto kptMatch1 = kptMatches.begin(); kptMatch1 != kptMatches.end(); kptMatch1++) {
        const cv::KeyPoint& kptCurr1 = kptsCurr[kptMatch1->trainIdx];
        const cv::KeyPoint& kptPrev1 = kptsPrev[kptMatch1->queryIdx];
        for (auto kptMatch2 = kptMatch1+1; kptMatch2 != kptMatches.end(); kptMatch2++) {
            const cv::KeyPoint& kptCurr2 = kptsCurr[kptMatch2->trainIdx];
            const cv::KeyPoint& kptPrev2 = kptsPrev[kptMatch2->queryIdx];
            double distanceCurr = cv::norm(kptCurr1.pt-kptCurr2.pt);
            double distancePrev = cv::norm(kptPrev1.pt-kptPrev2.pt);
            // check for minimal distance in current frame and avoid division by zero for distance in previous frame
            if (distanceCurr >= minDistance && distancePrev > std::numeric_limits<double>::epsilon()) {
                distanceRatios.push_back(distanceCurr/distancePrev);
            }
        }
    }
    cout << " - found " << distanceRatios.size() << " distance ratios" << endl;
    if (distanceRatios.size() != 0)
    {
        // find median distance ratio by sorting vector and finding median index
        sort(distanceRatios.begin(), distanceRatios.end());
        size_t medianIndex = distanceRatios.size()/2;
        double medDistanceRatio;
        if (distanceRatios.size() % 2 == 0)
        {
            medDistanceRatio = (distanceRatios[medianIndex-1]+distanceRatios[medianIndex])/2.0;
        }
        else
        {
            medDistanceRatio = distanceRatios[medianIndex];
        }
        // TTC as defined in course
        TTC = (1/frameRate)/(medDistanceRatio-1);
    }
    else
    {
        // TTC is not defined if not at least two keypoint matches were found sufficiently apart
        TTC = NAN;
    }
}


pcl::PointCloud<pcl::PointXYZ>::Ptr euclideanClustering(vector<LidarPoint>& lidarPoints) {
    // load lidar points into PCL library 
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (LidarPoint lidarPoint: lidarPoints)
    {
        inputCloud->push_back(pcl::PointXYZ((float)lidarPoint.x, (float)lidarPoint.y, (float)lidarPoint.z));
    }
    // for efficient search, use KdTree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr inputTree(new pcl::search::KdTree<pcl::PointXYZ>);
    inputTree->setInputCloud(inputCloud);
    // perform euclidean clustering
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclideanClustering;
    vector<pcl::PointIndices> clustersIndices;
    euclideanClustering.setInputCloud(inputCloud);
    euclideanClustering.setSearchMethod(inputTree);
    euclideanClustering.setMinClusterSize(4);
    euclideanClustering.setClusterTolerance(0.05);
    euclideanClustering.extract(clustersIndices);
    // construct new cloud containing only biggest cluster
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (const pcl::PointIndices& clusterIndices: clustersIndices)
    {
        for (size_t index: clusterIndices.indices)
        {
            outputCloud->points.push_back(inputCloud->points[index]);
        }
    }
    return outputCloud;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // remove outliers by performing euclidean clustering like we learned in the lidar course
    cout << "   lidar: before clustering: " << lidarPointsPrev.size() << " previous and " << lidarPointsCurr.size() << " current lidar points." << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr prevCloud = euclideanClustering(lidarPointsPrev);
    pcl::PointCloud<pcl::PointXYZ>::Ptr currCloud = euclideanClustering(lidarPointsCurr);
    cout << "   lidar: after clustering:  " << prevCloud->points.size() << " previous and " << currCloud->points.size() << " current lidar points." << endl;
    if (prevCloud->points.size() != 0 && currCloud->points.size() != 0)
    {
        // get nearest point from each point cloud
        double xMinPrev = prevCloud->points[0].x;
        for (const pcl::PointXYZ& point: prevCloud->points)
        {
            xMinPrev = point.x < xMinPrev ? point.x : xMinPrev;
        }
        double xMinCurr = currCloud->points[0].x;
        for (const pcl::PointXYZ& point: currCloud->points)
        {
            xMinCurr = point.x < xMinCurr ? point.x : xMinCurr;
        }
        // compute TTC
        TTC = xMinCurr * (1/frameRate) / (xMinPrev-xMinCurr);
    }
    else
    {
        // edge case: every thing has been smallclustered away
        TTC = NAN;
    }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // no multimap (as introduced in class), but map+vector instead, see
    // https://stackoverflow.com/questions/52074218/custom-compare-function-for-stdmultimap-when-keys-are-equal
    // from the "cv:DMatch"es, find the respective bounding boxes which contain the points and count them
    map<pair<int, int>, size_t> boxToBoxMatchCounts;
    for (const cv::DMatch& match: matches)
    {
        const cv::Point2f& prevFramePoint = prevFrame.keypoints[match.queryIdx].pt;
        const cv::Point2f& currFramePoint = currFrame.keypoints[match.trainIdx].pt;
        for (const BoundingBox& prevFrameBox: prevFrame.boundingBoxes)
        {
            if (prevFrameBox.roi.contains(prevFramePoint)) {
                for (const BoundingBox& currFrameBox: currFrame.boundingBoxes)
                {
                    boxToBoxMatchCounts[make_pair(prevFrameBox.boxID, currFrameBox.boxID)]++;
                }
            }
        }
    }
    // transform to a vector of triples (prevFrameBox.boxID, currFrameBoundingBox.boxID, matchCount computed above)
    // in order to more easily sort it
    vector<tuple<int, int, size_t>> boxToBoxMatchesTuples;
    for (const pair<pair<int, int>, size_t>& boxToBoxMatchCount: boxToBoxMatchCounts)
    {
        int prevFrameBoxID = boxToBoxMatchCount.first.first;
        int currFrameBoxID = boxToBoxMatchCount.first.second;
        size_t matchCount = boxToBoxMatchCount.second;
        boxToBoxMatchesTuples.push_back(make_tuple(prevFrameBoxID, currFrameBoxID, matchCount));
    }
    // sort it by the number of points that match between the bounding boxes
    sort(
            boxToBoxMatchesTuples.begin(), boxToBoxMatchesTuples.end(), 
            [](const tuple<int, int, size_t>& left, const tuple<int, int, size_t>& right) -> bool
            {  
                // sort by matchCount
                size_t matchCountLeft = get<2>(left);
                size_t matchCountRight = get<2>(right);
                return matchCountLeft > matchCountRight;
            }
        );
    // store best match in output "bbBestMatches" but make sure to only assign one match for every currFrameBoxID
    // since vector is sorted by best match we can traverse in descending order and memorize
    // which matchedPrevFrameBoxIDs we already saw
    bbBestMatches.clear();
    set<int> matchedPrevFrameBoxIDs;
    set<int> matchedCurrFrameBoxIDs;
    for (const tuple<int, int, size_t>& boxToBoxMatchTuple: boxToBoxMatchesTuples)
    {
        int prevFrameBoxID = get<0>(boxToBoxMatchTuple);
        int currFrameBoxID = get<1>(boxToBoxMatchTuple);
        size_t matchCount = get<2>(boxToBoxMatchTuple);
        if (matchedPrevFrameBoxIDs.find(prevFrameBoxID) == matchedPrevFrameBoxIDs.end())
        {
            matchedPrevFrameBoxIDs.insert(prevFrameBoxID);
            matchedCurrFrameBoxIDs.insert(currFrameBoxID);
            bbBestMatches.insert(make_pair(prevFrameBoxID, currFrameBoxID));
            cout << "   mapped BB " << prevFrameBoxID << " to " << currFrameBoxID << " (" << matchCount << " descriptor matches";
            for (const BoundingBox& currFrameBox: currFrame.boundingBoxes)
            {
                if (currFrameBox.boxID == currFrameBoxID && currFrameBox.lidarPoints.size() > 0)
                {
                    cout << " and " << currFrameBox.lidarPoints.size() << " lidar points";
                }
            }
            cout << ")." << endl;
        }
    }
    for (const BoundingBox& currFrameBox: currFrame.boundingBoxes)
    {
        if (matchedCurrFrameBoxIDs.find(currFrameBox.boxID) == matchedCurrFrameBoxIDs.end())
        {
            cout << "   BB " << currFrameBox.boxID << " was not matched (had " << currFrameBox.lidarPoints.size() << " lidar points)" << endl;
        }
    }
}
