// Object tracking algorithm using matchTemplate  
 

#include <opencv2/opencv.hpp>  

using namespace cv;
using namespace std;

// Global variables  （全局变量）
Rect box;
bool drawing_box = false;
bool gotBB = false;

// bounding box mouse callback （包围框鼠标回调） 
void mouseHandler(int event, int x, int y, int flags, void *param){
	switch (event){
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box){
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0){
			box.x += box.width;
			box.width *= -1;
		}
		if (box.height < 0){
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	}
}


// tracker: get search patches around the last tracking box,  
// and find the most similar one （跟踪：在最后一个跟踪框中找到搜索补丁，并找到最相似的一个） 
void tracking(Mat frame, Mat &model, Rect &trackBox)
{
	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);

	Rect searchWindow;
	searchWindow.width = trackBox.width * 3;
	searchWindow.height = trackBox.height * 3;
	searchWindow.x = trackBox.x + trackBox.width * 0.5 - searchWindow.width * 0.5;
	searchWindow.y = trackBox.y + trackBox.height * 0.5 - searchWindow.height * 0.5;
	searchWindow &= Rect(0, 0, frame.cols, frame.rows);

	Mat similarity;
	matchTemplate(gray(searchWindow), model, similarity, CV_TM_CCOEFF_NORMED);

	double mag_r;
	Point point;
	minMaxLoc(similarity, 0, &mag_r, 0, &point);
	trackBox.x = point.x + searchWindow.x;
	trackBox.y = point.y + searchWindow.y;
	model = gray(trackBox);
}

int main(int argc, char * argv[])
{
	VideoCapture capture;
	capture.open("david.mpg");//读取视频文件
	bool fromfile = true;
	//Init camera  
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		system("pause");
		return -1;
	}
	//Register mouse callback to draw the bounding box（寄存鼠标回调绘制包围框）  
	cvNamedWindow("Tracker", CV_WINDOW_AUTOSIZE);
	cvSetMouseCallback("Tracker", mouseHandler, NULL);

	Mat frame, model;
	capture >> frame;
	while (!gotBB)
	{
		if (!fromfile)
			capture >> frame;

		imshow("Tracker", frame);
		if (cvWaitKey(20) == 'q')
			return 1;
	}
	//Remove callback  
	cvSetMouseCallback("Tracker", NULL, NULL);

	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);
	model = gray(box);

	int frameCount = 0;

	while (1)
	{
		capture >> frame;
		if (frame.empty())
			return -1;
		double t = (double)cvGetTickCount();
		frameCount++;

		// tracking  
		tracking(frame, model, box);

		// show  
		stringstream buf;
		buf << frameCount;
		string num = buf.str();
		putText(frame, num, Point(20, 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
		rectangle(frame, box, Scalar(0, 0, 255), 3);
		imshow("Tracker", frame);


		t = (double)cvGetTickCount() - t;
		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;

		if (cvWaitKey(1) == 27)
			break;
	}

	system("pause");
	return 0;
}