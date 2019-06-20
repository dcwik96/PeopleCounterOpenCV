// opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/features2d.hpp"
// C
#include <stdio.h>
#include <stdlib.h>
// C++
#include <iostream>
#include <sstream>
#include <iomanip>
// CVUI
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "CVUI Test"
#define CHOOSE_FILE_WINDOW_NAME "Wybor filmu"

using namespace cv;
using namespace std;


VideoCapture capture;

Mat actual_frame;
Mat foreground_mask;
Mat start_window_frame = cv::Mat(1000, 1200, CV_8UC3);

Ptr<BackgroundSubtractor> substractor;

Point point1, point2;

bool filmZeSkrzyzowaniem = false;
bool filmZeStarowki = false;
bool checked = true;
bool pause = false;

int peopleCounter = 0;
int keyboard;
int drag = 0;
int distanceBetweenPosition = 10;

bool filterByColor = false;
bool filterByArea = true;
bool filterByCircularity = false;
bool filterByInertia = true;
bool filterByConvexity = false;

int thresholdStep = 10;
int minThreshold = 50;
int maxThreshold = 220;
int minRepeatability = 2;
int minDistanceBetweenBlobs = 10;
int blobColor = 0;
int minArea = 100;
int maxArea = 2500;

float minCircularity = 0.8f;
float maxCircularity = 1.0;
float minInertiaRatio = 0.01f;
float maxInertiaRatio = 0.1f;
float minConvexity = 0.1f;
float maxConvexity = 1.0;

void stopProgramWithSuccess() {
    capture.release();
    destroyAllWindows();
    exit(EXIT_SUCCESS);
}

void stopProgramWithFailure() {
    capture.release();
    destroyAllWindows();
    exit(EXIT_FAILURE);
}

void mouseHandler(int event, int x, int y, int flags, void *param) {
    if (event == EVENT_LBUTTONDOWN && !drag) {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
    }

    if (event == EVENT_MOUSEMOVE && drag) {
        /* mouse dragged. ROI being selected */
        Mat img1 = actual_frame.clone();
        point2 = Point(x, y);
        line(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", img1);
    }

    if (event == EVENT_LBUTTONUP && drag) {
        point2 = Point(x, y);
        drag = 0;
    }
}

// http://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
bool ccw(Point2f A, Point2f B, Point2f C) {
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

//Return true if line segments AB and CD intersect
bool intersecta(Point2f A, Point2f B, Point2f C, Point2f D) {
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D);
}

void startProgramWithFileChoose() {
    Mat choosingFileWindow = Mat(230, 250, CV_8UC3);

    cvui::init(CHOOSE_FILE_WINDOW_NAME);

    while (true) {
        choosingFileWindow = Scalar(49, 52, 49);

        cvui::beginColumn(choosingFileWindow, 10, 30, 100, 200, 10);
        cvui::text("Wybierz rodzaj filmu");
        if (cvui::checkbox("Film ze skrzyzowania", &filmZeSkrzyzowaniem)) {
            filmZeSkrzyzowaniem = true;
            filmZeStarowki = false;
        }
        if (cvui::checkbox("Film ze starowki", &filmZeStarowki)) {
            filmZeSkrzyzowaniem = false;
            filmZeStarowki = true;
        }
        cvui::text("WCISNIJ ENTER !");
        cvui::space(50);
        if (cvui::button("Zakoncz")) {
            stopProgramWithSuccess();
        }
        cvui::endColumn();

        cvui::update();
        imshow(CHOOSE_FILE_WINDOW_NAME, choosingFileWindow);

        if (cv::waitKey(20) == 13) {
            break;
        }
    }
    destroyAllWindows();
}

void showMenu() {
    start_window_frame = Scalar(49, 52, 49);

    cvui::beginColumn(start_window_frame, 10, 30, 100, 200, 10);

    cvui::text("Wybierz sposob oddzielania tla");
    if (cvui::button("MOG2"))
        substractor = createBackgroundSubtractorMOG2(); //MOG2
    if (cvui::button("KNN"))
        substractor = createBackgroundSubtractorKNN(); //KNN

    cvui::space(50);

    cvui::text("Czy usunac szumy?");
    cvui::checkbox("dilate/erode", &checked);
    cvui::text("Odleglosc pomiedzy pozycjami czlowieka");
    cvui::trackbar(300, &distanceBetweenPosition, 0, 50);

    cvui::space(50);
    cvui::printf("Ilosc osob = %d", peopleCounter);
    if (cvui::button("Wyzeruj licznik")) {
        peopleCounter = 0;
    }
    cvui::space(50);
    cvui::text("Ustaw parametry blob");

    cvui::text("Threshold step");
    cvui::trackbar(300, &thresholdStep, 0, 255);

    cvui::text("Min threshold");
    cvui::trackbar(300, &minThreshold, 0, 255);

    cvui::text("Max threshold");
    cvui::trackbar(300, &maxThreshold, 0, 255);

    cvui::text("Min powtarzalnosc");
    cvui::trackbar(300, &minRepeatability, 0, 10);

    cvui::text("Min dystans pomiedzy blobami");
    cvui::trackbar(300, &minDistanceBetweenBlobs, 0, 50);


    cvui::endColumn();


    cvui::beginColumn(start_window_frame, 350, 30, 100, 200, 10);

    cvui::text("Ustaw parametry blob");

    cvui::text("Filtrowanie po:");
    cvui::checkbox("kolorze", &filterByColor);
    cvui::text("Blob kolor");
    cvui::trackbar(300, &blobColor, 0, 255);
    cvui::space(20);

    cvui::checkbox("strefie obiektu", &filterByArea);
    cvui::text("Min strefa");
    cvui::trackbar(300, &minArea, 0, 3500);
    cvui::text("Max strefa");
    cvui::trackbar(300, &maxArea, 0, 3500);
    cvui::space(20);

    cvui::checkbox("kolistosc", &filterByCircularity);
    cvui::text("Min kolistosc");
    cvui::trackbar(300, &minCircularity, 0.f, 1.f);
    cvui::text("Max kolistosc");
    cvui::trackbar(300, &maxCircularity, 0.f, 1.f);
    cvui::space(20);

    cvui::checkbox("inertia (bezwladnosc)", &filterByInertia);
    cvui::text("Min inertia");
    cvui::trackbar(300, &minInertiaRatio, 0.f, 1.f);
    cvui::text("Max inertia");
    cvui::trackbar(300, &maxInertiaRatio, 0.f, 1.f);
    cvui::space(20);

    cvui::checkbox("wygiecie", &filterByConvexity);
    cvui::text("Min wygiecie");
    cvui::trackbar(300, &minConvexity, 0.f, 1.f);
    cvui::text("Max inertia");
    cvui::trackbar(300, &maxConvexity, 0.f, 1.f);

    cvui::endColumn();


    cvui::beginColumn(start_window_frame, 800, 30, 100, 200, 10);
    if (cvui::button(350, 100, "PAUSE")) {
        pause = true;
    }
    if (cvui::button(350, 100, "PLAY")) {
        pause = false;
    }
    cvui::endColumn();


    cvui::update();
    cv::imshow(WINDOW_NAME, start_window_frame);
}

void setLineOnVideo() {
    while (true) {
        setMouseCallback("Frame", mouseHandler, NULL);
        Mat img1 = actual_frame.clone();
        putText(img1, "ZAZNACZ LINIE", Point(10, 30),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(250, 0, 250), 1, LINE_AA);
        putText(img1, "Potwierdz ENTER", Point(10, 60),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(250, 0, 250), 1, LINE_AA);
        line(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", img1);
        if (cv::waitKey(20) == 13) {
            break;
        }
    }
}

void loadVideo() {
    if (filmZeStarowki)
        capture.open("/home/dcwik/Pobrane/[10 sec fragment, HDconvert.com] TownCentreXVID.avi");
    else if (filmZeSkrzyzowaniem)
        capture.open("/home/dcwik/Pobrane/Video_003.avi");
    else {
        printf("Brak poprawnego wyboru filmu");
        stopProgramWithFailure();
    }
}

int main() {
    startProgramWithFileChoose();
    loadVideo();

    substractor = createBackgroundSubtractorMOG2();//MOG2

    if (!capture.isOpened()) {
        printf("Nie mozna otworzyc wideo");
        getchar();
        stopProgramWithFailure();
    }

    if (!pause)
        capture >> actual_frame;

    if (actual_frame.empty()) {
        printf("Nie mozna odczytac kolejnej klatki filmu");
        getchar();
        stopProgramWithFailure();
    }

    fflush(stdout);

    setLineOnVideo();

    Mat im_detecciones;
    vector<cv::KeyPoint> personas;
    vector<cv::Point2f> prevPersonas;
    Point2f persona;

    cvui::init(WINDOW_NAME);

    while ((char) keyboard != 27) {
        showMenu();
        SimpleBlobDetector::Params params;
        params.thresholdStep = thresholdStep;
        params.minThreshold = minThreshold;
        params.maxThreshold = maxThreshold;
        params.minRepeatability = minRepeatability;
        params.minDistBetweenBlobs = minDistanceBetweenBlobs;

        params.filterByColor = filterByColor;
        params.blobColor = blobColor;

        params.filterByArea = filterByArea;
        params.minArea = minArea;
        params.maxArea = maxArea;

        params.filterByCircularity = filterByCircularity;
        params.minCircularity = minCircularity;
        params.maxCircularity = maxCircularity;

        params.filterByInertia = filterByInertia;
        params.minInertiaRatio = minInertiaRatio;
        params.maxInertiaRatio = maxInertiaRatio;

        params.filterByConvexity = filterByConvexity;
        params.minConvexity = minConvexity;
        params.maxConvexity = maxConvexity;

        Ptr<cv::SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        if (!pause)
            capture >> actual_frame;

        if (actual_frame.empty()) {
            printf("Film sie skonczyl");
            capture.release();
            destroyAllWindows();
            main();
        }

        substractor->apply(actual_frame, foreground_mask);

        if (checked) {
            dilate(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            erode(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            erode(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            dilate(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
        }

        line(actual_frame, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);


        // https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        KeyPoint::convert(personas, prevPersonas);
        detector->detect(foreground_mask, personas);
        drawKeypoints(actual_frame, personas, im_detecciones, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (prevPersonas.size() >= personas.size()) {
            int i = 0;
            for (auto &blobIterator : personas) {
                printf("%f  ;  %f \n", persona.x, persona.y);
                persona = Point2f(blobIterator.pt.x, blobIterator.pt.y);
                double odlegloscMiedzyPozycjami = norm(persona - prevPersonas[i]);
                if (odlegloscMiedzyPozycjami < distanceBetweenPosition) {
                    line(im_detecciones, persona, prevPersonas[i], CV_RGB(0, 255, 0), 3, 8, 0);
                    if (intersecta(persona, prevPersonas[i], point1, point2)) {
                        peopleCounter++;
                        printf("%d\n", peopleCounter);
                    }
                }
                i++;
            }
        }

        imshow("keypoints", im_detecciones);
        imshow("Frame", actual_frame);
        imshow("Foreground", foreground_mask);
        if (cv::waitKey(20) == 27) {
            break;
        }
    }

    capture.release();
    destroyAllWindows();
    return EXIT_SUCCESS;
}

// opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/features2d.hpp"
// C
#include <stdio.h>
#include <stdlib.h>
// C++
#include <iostream>
#include <sstream>
#include <iomanip>
// CVUI
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "CVUI Test"
#define CHOOSE_FILE_WINDOW_NAME "Wybor filmu"

using namespace cv;
using namespace std;


VideoCapture capture;

Mat actual_frame;
Mat foreground_mask;
Mat start_window_frame = cv::Mat(1000, 1200, CV_8UC3);

Ptr<BackgroundSubtractor> substractor;

Point point1, point2;

bool filmZeSkrzyzowaniem = false;
bool filmZeStarowki = false;
bool checked = true;
bool pause = false;

int peopleCounter = 0;
int keyboard;
int drag = 0;
int distanceBetweenPosition = 10;

bool filterByColor = false;
bool filterByArea = true;
bool filterByCircularity = false;
bool filterByInertia = true;
bool filterByConvexity = false;

int thresholdStep = 10;
int minThreshold = 50;
int maxThreshold = 220;
int minRepeatability = 2;
int minDistanceBetweenBlobs = 10;
int blobColor = 0;
int minArea = 100;
int maxArea = 2500;

float minCircularity = 0.8f;
float maxCircularity = 1.0;
float minInertiaRatio = 0.01f;
float maxInertiaRatio = 0.1f;
float minConvexity = 0.1f;
float maxConvexity = 1.0;

void stopProgramWithSuccess() {
    capture.release();
    destroyAllWindows();
    exit(EXIT_SUCCESS);
}

void stopProgramWithFailure() {
    capture.release();
    destroyAllWindows();
    exit(EXIT_FAILURE);
}

void mouseHandler(int event, int x, int y, int flags, void *param) {
    if (event == EVENT_LBUTTONDOWN && !drag) {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
    }

    if (event == EVENT_MOUSEMOVE && drag) {
        /* mouse dragged. ROI being selected */
        Mat img1 = actual_frame.clone();
        point2 = Point(x, y);
        line(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", img1);
    }

    if (event == EVENT_LBUTTONUP && drag) {
        point2 = Point(x, y);
        drag = 0;
    }
}

// http://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
bool ccw(Point2f A, Point2f B, Point2f C) {
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

//Return true if line segments AB and CD intersect
bool intersecta(Point2f A, Point2f B, Point2f C, Point2f D) {
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D);
}

void startProgramWithFileChoose() {
    Mat choosingFileWindow = Mat(230, 250, CV_8UC3);

    cvui::init(CHOOSE_FILE_WINDOW_NAME);

    while (true) {
        choosingFileWindow = Scalar(49, 52, 49);

        cvui::beginColumn(choosingFileWindow, 10, 30, 100, 200, 10);
        cvui::text("Wybierz rodzaj filmu");
        if (cvui::checkbox("Film ze skrzyzowania", &filmZeSkrzyzowaniem)) {
            filmZeSkrzyzowaniem = true;
            filmZeStarowki = false;
        }
        if (cvui::checkbox("Film ze starowki", &filmZeStarowki)) {
            filmZeSkrzyzowaniem = false;
            filmZeStarowki = true;
        }
        cvui::text("WCISNIJ ENTER !");
        cvui::space(50);
        if (cvui::button("Zakoncz")) {
            stopProgramWithSuccess();
        }
        cvui::endColumn();

        cvui::update();
        imshow(CHOOSE_FILE_WINDOW_NAME, choosingFileWindow);

        if (cv::waitKey(20) == 13) {
            break;
        }
    }
    destroyAllWindows();
}

void showMenu() {
    start_window_frame = Scalar(49, 52, 49);

    cvui::beginColumn(start_window_frame, 10, 30, 100, 200, 10);

    cvui::text("Wybierz sposob oddzielania tla");
    if (cvui::button("MOG2"))
        substractor = createBackgroundSubtractorMOG2(); //MOG2
    if (cvui::button("KNN"))
        substractor = createBackgroundSubtractorKNN(); //KNN

    cvui::space(50);

    cvui::text("Czy usunac szumy?");
    cvui::checkbox("dilate/erode", &checked);
    cvui::text("Odleglosc pomiedzy pozycjami czlowieka");
    cvui::trackbar(300, &distanceBetweenPosition, 0, 50);

    cvui::space(50);
    cvui::printf("Ilosc osob = %d", peopleCounter);
    if (cvui::button("Wyzeruj licznik")) {
        peopleCounter = 0;
    }
    cvui::space(50);
    cvui::text("Ustaw parametry blob");

    cvui::text("Threshold step");
    cvui::trackbar(300, &thresholdStep, 0, 255);

    cvui::text("Min threshold");
    cvui::trackbar(300, &minThreshold, 0, 255);

    cvui::text("Max threshold");
    cvui::trackbar(300, &maxThreshold, 0, 255);

    cvui::text("Min powtarzalnosc");
    cvui::trackbar(300, &minRepeatability, 0, 10);

    cvui::text("Min dystans pomiedzy blobami");
    cvui::trackbar(300, &minDistanceBetweenBlobs, 0, 50);


    cvui::endColumn();


    cvui::beginColumn(start_window_frame, 350, 30, 100, 200, 10);

    cvui::text("Ustaw parametry blob");

    cvui::text("Filtrowanie po:");
    cvui::checkbox("kolorze", &filterByColor);
    cvui::text("Blob kolor");
    cvui::trackbar(300, &blobColor, 0, 255);
    cvui::space(20);

    cvui::checkbox("strefie obiektu", &filterByArea);
    cvui::text("Min strefa");
    cvui::trackbar(300, &minArea, 0, 3500);
    cvui::text("Max strefa");
    cvui::trackbar(300, &maxArea, 0, 3500);
    cvui::space(20);

    cvui::checkbox("kolistosc", &filterByCircularity);
    cvui::text("Min kolistosc");
    cvui::trackbar(300, &minCircularity, 0.f, 1.f);
    cvui::text("Max kolistosc");
    cvui::trackbar(300, &maxCircularity, 0.f, 1.f);
    cvui::space(20);

    cvui::checkbox("inertia (bezwladnosc)", &filterByInertia);
    cvui::text("Min inertia");
    cvui::trackbar(300, &minInertiaRatio, 0.f, 1.f);
    cvui::text("Max inertia");
    cvui::trackbar(300, &maxInertiaRatio, 0.f, 1.f);
    cvui::space(20);

    cvui::checkbox("wygiecie", &filterByConvexity);
    cvui::text("Min wygiecie");
    cvui::trackbar(300, &minConvexity, 0.f, 1.f);
    cvui::text("Max inertia");
    cvui::trackbar(300, &maxConvexity, 0.f, 1.f);

    cvui::endColumn();


    cvui::beginColumn(start_window_frame, 800, 30, 100, 200, 10);
    if (cvui::button(350, 100, "PAUSE")) {
        pause = true;
    }
    if (cvui::button(350, 100, "PLAY")) {
        pause = false;
    }
    cvui::endColumn();


    cvui::update();
    cv::imshow(WINDOW_NAME, start_window_frame);
}

void setLineOnVideo() {
    while (true) {
        setMouseCallback("Frame", mouseHandler, NULL);
        Mat img1 = actual_frame.clone();
        putText(img1, "ZAZNACZ LINIE", Point(10, 30),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(250, 0, 250), 1, LINE_AA);
        putText(img1, "Potwierdz ENTER", Point(10, 60),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(250, 0, 250), 1, LINE_AA);
        line(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", img1);
        if (cv::waitKey(20) == 13) {
            break;
        }
    }
}

void loadVideo() {
    if (filmZeStarowki)
        capture.open("/home/dcwik/Pobrane/[10 sec fragment, HDconvert.com] TownCentreXVID.avi");
    else if (filmZeSkrzyzowaniem)
        capture.open("/home/dcwik/Pobrane/Video_003.avi");
    else {
        printf("Brak poprawnego wyboru filmu");
        stopProgramWithFailure();
    }
}

int main() {
    startProgramWithFileChoose();
    loadVideo();

    substractor = createBackgroundSubtractorMOG2();//MOG2

    if (!capture.isOpened()) {
        printf("Nie mozna otworzyc wideo");
        getchar();
        stopProgramWithFailure();
    }

    if (!pause)
        capture >> actual_frame;

    if (actual_frame.empty()) {
        printf("Nie mozna odczytac kolejnej klatki filmu");
        getchar();
        stopProgramWithFailure();
    }

    fflush(stdout);

    setLineOnVideo();

    Mat im_detecciones;
    vector<cv::KeyPoint> personas;
    vector<cv::Point2f> prevPersonas;
    Point2f persona;

    cvui::init(WINDOW_NAME);

    while ((char) keyboard != 27) {
        showMenu();
        SimpleBlobDetector::Params params;
        params.thresholdStep = thresholdStep;
        params.minThreshold = minThreshold;
        params.maxThreshold = maxThreshold;
        params.minRepeatability = minRepeatability;
        params.minDistBetweenBlobs = minDistanceBetweenBlobs;

        params.filterByColor = filterByColor;
        params.blobColor = blobColor;

        params.filterByArea = filterByArea;
        params.minArea = minArea;
        params.maxArea = maxArea;

        params.filterByCircularity = filterByCircularity;
        params.minCircularity = minCircularity;
        params.maxCircularity = maxCircularity;

        params.filterByInertia = filterByInertia;
        params.minInertiaRatio = minInertiaRatio;
        params.maxInertiaRatio = maxInertiaRatio;

        params.filterByConvexity = filterByConvexity;
        params.minConvexity = minConvexity;
        params.maxConvexity = maxConvexity;

        Ptr<cv::SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        if (!pause)
            capture >> actual_frame;

        if (actual_frame.empty()) {
            printf("Film sie skonczyl");
            capture.release();
            destroyAllWindows();
            main();
        }

        substractor->apply(actual_frame, foreground_mask);

        if (checked) {
            dilate(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            erode(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            erode(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            dilate(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
        }

        line(actual_frame, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);


        // https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        KeyPoint::convert(personas, prevPersonas);
        detector->detect(foreground_mask, personas);
        drawKeypoints(actual_frame, personas, im_detecciones, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (prevPersonas.size() >= personas.size()) {
            int i = 0;
            for (auto &blobIterator : personas) {
                printf("%f  ;  %f \n", persona.x, persona.y);
                persona = Point2f(blobIterator.pt.x, blobIterator.pt.y);
                double odlegloscMiedzyPozycjami = norm(persona - prevPersonas[i]);
                if (odlegloscMiedzyPozycjami < distanceBetweenPosition) {
                    line(im_detecciones, persona, prevPersonas[i], CV_RGB(0, 255, 0), 3, 8, 0);
                    if (intersecta(persona, prevPersonas[i], point1, point2)) {
                        peopleCounter++;
                        printf("%d\n", peopleCounter);
                    }
                }
                i++;
            }
        }

        imshow("keypoints", im_detecciones);
        imshow("Frame", actual_frame);
        imshow("Foreground", foreground_mask);
        if (cv::waitKey(20) == 27) {
            break;
        }
    }

    capture.release();
    destroyAllWindows();
    return EXIT_SUCCESS;
}

// opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/features2d.hpp"
// C
#include <stdio.h>
#include <stdlib.h>
// C++
#include <iostream>
#include <sstream>
#include <iomanip>
// CVUI
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#define WINDOW_NAME "CVUI Test"
#define CHOOSE_FILE_WINDOW_NAME "Wybor filmu"

using namespace cv;
using namespace std;


VideoCapture capture;

Mat actual_frame;
Mat foreground_mask;
Mat start_window_frame = cv::Mat(1000, 1200, CV_8UC3);

Ptr<BackgroundSubtractor> substractor;

Point point1, point2;

bool filmZeSkrzyzowaniem = false;
bool filmZeStarowki = false;
bool checked = true;
bool pause = false;

int peopleCounter = 0;
int keyboard;
int drag = 0;
int distanceBetweenPosition = 10;

bool filterByColor = false;
bool filterByArea = true;
bool filterByCircularity = false;
bool filterByInertia = true;
bool filterByConvexity = false;

int thresholdStep = 10;
int minThreshold = 50;
int maxThreshold = 220;
int minRepeatability = 2;
int minDistanceBetweenBlobs = 10;
int blobColor = 0;
int minArea = 100;
int maxArea = 2500;

float minCircularity = 0.8f;
float maxCircularity = 1.0;
float minInertiaRatio = 0.01f;
float maxInertiaRatio = 0.1f;
float minConvexity = 0.1f;
float maxConvexity = 1.0;

void stopProgramWithSuccess() {
    capture.release();
    destroyAllWindows();
    exit(EXIT_SUCCESS);
}

void stopProgramWithFailure() {
    capture.release();
    destroyAllWindows();
    exit(EXIT_FAILURE);
}

void mouseHandler(int event, int x, int y, int flags, void *param) {
    if (event == EVENT_LBUTTONDOWN && !drag) {
        /* left button clicked. ROI selection begins */
        point1 = Point(x, y);
        drag = 1;
    }

    if (event == EVENT_MOUSEMOVE && drag) {
        /* mouse dragged. ROI being selected */
        Mat img1 = actual_frame.clone();
        point2 = Point(x, y);
        line(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", img1);
    }

    if (event == EVENT_LBUTTONUP && drag) {
        point2 = Point(x, y);
        drag = 0;
    }
}

// http://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
bool ccw(Point2f A, Point2f B, Point2f C) {
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x);
}

//Return true if line segments AB and CD intersect
bool intersecta(Point2f A, Point2f B, Point2f C, Point2f D) {
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D);
}

void startProgramWithFileChoose() {
    Mat choosingFileWindow = Mat(230, 250, CV_8UC3);

    cvui::init(CHOOSE_FILE_WINDOW_NAME);

    while (true) {
        choosingFileWindow = Scalar(49, 52, 49);

        cvui::beginColumn(choosingFileWindow, 10, 30, 100, 200, 10);
        cvui::text("Wybierz rodzaj filmu");
        if (cvui::checkbox("Film ze skrzyzowania", &filmZeSkrzyzowaniem)) {
            filmZeSkrzyzowaniem = true;
            filmZeStarowki = false;
        }
        if (cvui::checkbox("Film ze starowki", &filmZeStarowki)) {
            filmZeSkrzyzowaniem = false;
            filmZeStarowki = true;
        }
        cvui::text("WCISNIJ ENTER !");
        cvui::space(50);
        if (cvui::button("Zakoncz")) {
            stopProgramWithSuccess();
        }
        cvui::endColumn();

        cvui::update();
        imshow(CHOOSE_FILE_WINDOW_NAME, choosingFileWindow);

        if (cv::waitKey(20) == 13) {
            break;
        }
    }
    destroyAllWindows();
}

void showMenu() {
    start_window_frame = Scalar(49, 52, 49);

    cvui::beginColumn(start_window_frame, 10, 30, 100, 200, 10);

    cvui::text("Wybierz sposob oddzielania tla");
    if (cvui::button("MOG2"))
        substractor = createBackgroundSubtractorMOG2(); //MOG2
    if (cvui::button("KNN"))
        substractor = createBackgroundSubtractorKNN(); //KNN

    cvui::space(50);

    cvui::text("Czy usunac szumy?");
    cvui::checkbox("dilate/erode", &checked);
    cvui::text("Odleglosc pomiedzy pozycjami czlowieka");
    cvui::trackbar(300, &distanceBetweenPosition, 0, 50);

    cvui::space(50);
    cvui::printf("Ilosc osob = %d", peopleCounter);
    if (cvui::button("Wyzeruj licznik")) {
        peopleCounter = 0;
    }
    cvui::space(50);
    cvui::text("Ustaw parametry blob");

    cvui::text("Threshold step");
    cvui::trackbar(300, &thresholdStep, 0, 255);

    cvui::text("Min threshold");
    cvui::trackbar(300, &minThreshold, 0, 255);

    cvui::text("Max threshold");
    cvui::trackbar(300, &maxThreshold, 0, 255);

    cvui::text("Min powtarzalnosc");
    cvui::trackbar(300, &minRepeatability, 0, 10);

    cvui::text("Min dystans pomiedzy blobami");
    cvui::trackbar(300, &minDistanceBetweenBlobs, 0, 50);


    cvui::endColumn();


    cvui::beginColumn(start_window_frame, 350, 30, 100, 200, 10);

    cvui::text("Ustaw parametry blob");

    cvui::text("Filtrowanie po:");
    cvui::checkbox("kolorze", &filterByColor);
    cvui::text("Blob kolor");
    cvui::trackbar(300, &blobColor, 0, 255);
    cvui::space(20);

    cvui::checkbox("strefie obiektu", &filterByArea);
    cvui::text("Min strefa");
    cvui::trackbar(300, &minArea, 0, 3500);
    cvui::text("Max strefa");
    cvui::trackbar(300, &maxArea, 0, 3500);
    cvui::space(20);

    cvui::checkbox("kolistosc", &filterByCircularity);
    cvui::text("Min kolistosc");
    cvui::trackbar(300, &minCircularity, 0.f, 1.f);
    cvui::text("Max kolistosc");
    cvui::trackbar(300, &maxCircularity, 0.f, 1.f);
    cvui::space(20);

    cvui::checkbox("inertia (bezwladnosc)", &filterByInertia);
    cvui::text("Min inertia");
    cvui::trackbar(300, &minInertiaRatio, 0.f, 1.f);
    cvui::text("Max inertia");
    cvui::trackbar(300, &maxInertiaRatio, 0.f, 1.f);
    cvui::space(20);

    cvui::checkbox("wygiecie", &filterByConvexity);
    cvui::text("Min wygiecie");
    cvui::trackbar(300, &minConvexity, 0.f, 1.f);
    cvui::text("Max inertia");
    cvui::trackbar(300, &maxConvexity, 0.f, 1.f);

    cvui::endColumn();


    cvui::beginColumn(start_window_frame, 800, 30, 100, 200, 10);
    if (cvui::button(350, 100, "PAUSE")) {
        pause = true;
    }
    if (cvui::button(350, 100, "PLAY")) {
        pause = false;
    }
    cvui::endColumn();


    cvui::update();
    cv::imshow(WINDOW_NAME, start_window_frame);
}

void setLineOnVideo() {
    while (true) {
        setMouseCallback("Frame", mouseHandler, NULL);
        Mat img1 = actual_frame.clone();
        putText(img1, "ZAZNACZ LINIE", Point(10, 30),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(250, 0, 250), 1, LINE_AA);
        putText(img1, "Potwierdz ENTER", Point(10, 60),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, Scalar(250, 0, 250), 1, LINE_AA);
        line(img1, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);
        imshow("Frame", img1);
        if (cv::waitKey(20) == 13) {
            break;
        }
    }
}

void loadVideo() {
    if (filmZeStarowki)
        capture.open("/home/dcwik/Pobrane/[10 sec fragment, HDconvert.com] TownCentreXVID.avi");
    else if (filmZeSkrzyzowaniem)
        capture.open("/home/dcwik/Pobrane/Video_003.avi");
    else {
        printf("Brak poprawnego wyboru filmu");
        stopProgramWithFailure();
    }
}

int main() {
    startProgramWithFileChoose();
    loadVideo();

    substractor = createBackgroundSubtractorMOG2();//MOG2

    if (!capture.isOpened()) {
        printf("Nie mozna otworzyc wideo");
        getchar();
        stopProgramWithFailure();
    }

    if (!pause)
        capture >> actual_frame;

    if (actual_frame.empty()) {
        printf("Nie mozna odczytac kolejnej klatki filmu");
        getchar();
        stopProgramWithFailure();
    }

    fflush(stdout);

    setLineOnVideo();

    Mat im_detecciones;
    vector<cv::KeyPoint> personas;
    vector<cv::Point2f> prevPersonas;
    Point2f persona;

    cvui::init(WINDOW_NAME);

    while ((char) keyboard != 27) {
        showMenu();
        SimpleBlobDetector::Params params;
        params.thresholdStep = thresholdStep;
        params.minThreshold = minThreshold;
        params.maxThreshold = maxThreshold;
        params.minRepeatability = minRepeatability;
        params.minDistBetweenBlobs = minDistanceBetweenBlobs;

        params.filterByColor = filterByColor;
        params.blobColor = blobColor;

        params.filterByArea = filterByArea;
        params.minArea = minArea;
        params.maxArea = maxArea;

        params.filterByCircularity = filterByCircularity;
        params.minCircularity = minCircularity;
        params.maxCircularity = maxCircularity;

        params.filterByInertia = filterByInertia;
        params.minInertiaRatio = minInertiaRatio;
        params.maxInertiaRatio = maxInertiaRatio;

        params.filterByConvexity = filterByConvexity;
        params.minConvexity = minConvexity;
        params.maxConvexity = maxConvexity;

        Ptr<cv::SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

        if (!pause)
            capture >> actual_frame;

        if (actual_frame.empty()) {
            printf("Film sie skonczyl");
            capture.release();
            destroyAllWindows();
            main();
        }

        substractor->apply(actual_frame, foreground_mask);

        if (checked) {
            dilate(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            erode(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            erode(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
            dilate(foreground_mask, foreground_mask, getStructuringElement(MORPH_ELLIPSE, Size(2, 2)));
        }

        line(actual_frame, point1, point2, CV_RGB(255, 0, 0), 2, 8, 0);


        // https://www.learnopencv.com/blob-detection-using-opencv-python-c/
        KeyPoint::convert(personas, prevPersonas);
        detector->detect(foreground_mask, personas);
        drawKeypoints(actual_frame, personas, im_detecciones, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        if (prevPersonas.size() >= personas.size()) {
            int i = 0;
            for (auto &blobIterator : personas) {
                printf("%f  ;  %f \n", persona.x, persona.y);
                persona = Point2f(blobIterator.pt.x, blobIterator.pt.y);
                double odlegloscMiedzyPozycjami = norm(persona - prevPersonas[i]);
                if (odlegloscMiedzyPozycjami < distanceBetweenPosition) {
                    line(im_detecciones, persona, prevPersonas[i], CV_RGB(0, 255, 0), 3, 8, 0);
                    if (intersecta(persona, prevPersonas[i], point1, point2)) {
                        peopleCounter++;
                        printf("%d\n", peopleCounter);
                    }
                }
                i++;
            }
        }

        imshow("keypoints", im_detecciones);
        imshow("Frame", actual_frame);
        imshow("Foreground", foreground_mask);
        if (cv::waitKey(20) == 27) {
            break;
        }
    }

    capture.release();
    destroyAllWindows();
    return EXIT_SUCCESS;
}

