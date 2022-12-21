#include "ValFaceCapture.hpp"
#include "ValFaceCaptureImpl.hpp"
#include "../ValSharedObjects/ValSharedFrame/ValSharedFrame.hpp"
#include "../ValSharedObjects/ValSharedFrame/ValSharedFrameData.hpp"

#include <opencv2/core/mat.hpp>
extern "C" {
    #include <libavcodec/avcodec.h>
}

#include "facedetectcnn.h"

#include "val-frs-variables.hpp"
#include "performance-timer.hpp"
#include "resource-remover.hpp"
#include "napi-translate.hpp"

#include "impl-cpp-include.h"

std::shared_ptr<spdlog::logger> ValFaceCapture::Impl::sharedLogger = CreateLogger(
    logTitle,
    spdlog::level::info);

void ValFaceCapture::Impl::SetLogLevel(const spdlog::level::level_enum& logLevel) {
    this->logger->set_level(logLevel);
}

ValFaceCapture::Impl::Impl() :
    model(ValFRSGlobal::SharedModel()),
    logger(sharedLogger) {}

ValFaceCapture::Impl::Impl(const Ctor_Info& info) :
    ValFaceCapture::Impl::Impl() {}

inline std::vector<std::unique_ptr<ValSharedCapturedFaceData>> ValFaceCapture::Impl::detect(const std::shared_ptr<cv::Mat>& data,
    int maxCaptureFaces, float confidenceScore,
    int minFaceLength,
    float maxYawAngle, float maxPitchAngle) {

    auto faces = model->detect(*data,
        maxCaptureFaces, confidenceScore, minFaceLength, maxYawAngle, maxPitchAngle);

    /// translate to ValSharedCapturedFaceData
    std::vector<std::unique_ptr<ValSharedCapturedFaceData>> rtn;
    rtn.reserve(faces.size());
    for (auto& face : faces) {
        rtn.emplace_back( std::make_unique<ValSharedCapturedFaceData>(data, std::move(face)) );
    }
    return rtn;
}

inline std::vector<std::unique_ptr<ValSharedCapturedFaceData>> ValFaceCapture::Impl::detect(const std::shared_ptr<AVFrame>& data,
    int maxCaptureFaces, float confidenceScore,
    int minFaceLength,
    float maxYawAngle, float maxPitchAngle) {

    auto faces = model->detect(*data,
        maxCaptureFaces, confidenceScore, minFaceLength, maxYawAngle, maxPitchAngle);

    /// translate to ValSharedCapturedFaceData
    std::vector<std::unique_ptr<ValSharedCapturedFaceData>> rtn;
    rtn.reserve(faces.size());
    for (unsigned i=0; i<faces.size(); ++i) {
        auto& face = faces[i];
        rtn.emplace_back( std::make_unique<ValSharedCapturedFaceData>(data, std::move(face)) );
    }

    return rtn;
}

inline std::vector<std::unique_ptr<ValSharedCapturedFaceData>> ValFaceCapture::Impl::detect(const ValSharedCapturedFaceData& data,
    int maxCaptureFaces, float confidenceScore,
    int minFaceLength,
    float maxYawAngle, float maxPitchAngle) {

    /// handle both Buffer & AVFrame
    if (data.IsMat()) {
        return detect(data.mat,
            maxCaptureFaces, confidenceScore, minFaceLength, maxYawAngle, maxPitchAngle);

    } else {
        return detect(data.frame,
            maxCaptureFaces, confidenceScore, minFaceLength, maxYawAngle, maxPitchAngle);
    }
}

std::vector<std::unique_ptr<ValSharedCapturedFaceData>> ValFaceCapture::Impl::Detect(const Detect_Info& info) {
    constexpr int redetect_length = ValFRSVariables::RedetectLength;
    constexpr int detect_min_length = ValFRSVariables::DetectMinLength;

    PerformanceTimer pt(*this->logger, "Face Detect");

    auto datas = this->detect(*info.dataptr.get(), info.captureFaces, info.confidenceScore, info.minFaceLength); /// ignore these with thumbnail pic. maxYawAngle, maxPitchAngle);
    std::vector<std::unique_ptr<ValSharedCapturedFaceData>> finaldatas;
    finaldatas.reserve(datas.size());

    /// perform re-detect
    for (auto& data : datas) {
        auto& face = *data->face;

        /// test ROI match percent
        if (info.rois && info.rois->size() > 0) {
            VPolygon facePoly, inter;
            facePoly.clear();
            facePoly.add(cv::Point( face.x, face.y ));
            facePoly.add(cv::Point( face.x + face.w, face.y ));
            facePoly.add(cv::Point( face.x + face.w, face.y + face.h ));
            facePoly.add(cv::Point( face.x, face.y + face.h ));

            bool isIntersect = false;
            for (auto it = info.rois->cbegin(); it != info.rois->cend(); ++it) {
                intersectPolygon(facePoly, *it, inter);
                if ((inter.area() / facePoly.area()) >= ValFRSVariables::ROIIntersectRatio) { isIntersect = true; break; }
            }
            if (!isIntersect) continue;
        }

        /// test face size for re-detect or not
        if (
            /// case 1: face too small for re-detect
            (face.w < detect_min_length || face.h < detect_min_length) ||
            /// case 2: face large enough
            (face.w > redetect_length || face.h > redetect_length)
            ) {

            /// handle right here: maxYawAngle, maxPitchAngle
            if (
                face.yaw <= info.maxYawAngle &&
                face.pitch <= info.maxPitchAngle) {
                    finaldatas.emplace_back(std::move(data));
                }
            continue;
        }

        auto mat = data->getMat(EFaceSnapshotType::Image, true);
        auto rtnvect = this->detect(std::move(mat), 1, info.confidenceScore, info.minFaceLength, info.maxYawAngle, info.maxPitchAngle);
        /// handle exception: detect no face in larger area
        if (rtnvect.size() == 0) continue;
        auto& newface = *rtnvect[0]->face;

        /// translate 5 points
        for (int j=0; j<10; j++) {
            if (j%2 == 0) newface.lm[j] += face.x;
            else newface.lm[j] += face.y;
        }
        /// translate x & y
        newface.x += face.x;
        newface.y += face.y;
        newface.orix += face.x;
        newface.oriy += face.y;
        /// extend the area without scale
        // model->doTransformFaceRectNoResize(newface, data->size());
        memcpy(&face, &newface, sizeof(FaceRect));

        finaldatas.emplace_back(std::move(data));
    }
    return finaldatas;
}

ValFaceCapture::Impl::Ctor_Info::Ctor_Info() {}
ValFaceCapture::Impl::Ctor_Info::Ctor_Info(const Napi::CallbackInfo& info) {
    /// do nothing
}

std::unique_ptr<ValSharedCapturedFaceData> ValFaceCapture::Impl::Detect_Info::FromBufferOrFrame(const Napi::Value& bufferOrFrame) {
    if (bufferOrFrame.IsBuffer()) {
        auto oframe = bufferOrFrame.As<Napi::Buffer<char>>();
        auto mat = makeMatFromBuffer(oframe);
        return std::make_unique<ValSharedCapturedFaceData>(ValSharedCapturedFaceData(std::move(mat)));

    } else {
        auto oframe = bufferOrFrame.As<Napi::Object>();
        ValSharedFrame* sharedframe = ValSharedFrame::Unwrap(oframe);
        std::shared_ptr<ValSharedFrameData> framedata = sharedframe->GetData();
        return std::make_unique<ValSharedCapturedFaceData>(ValSharedCapturedFaceData(framedata->frame));
    }
}

void ValFaceCapture::Impl::Detect_Info::Detect_Config_Info() {
    this->confidenceScore = ValFRSVariables::ConfidenceScore;
    this->captureFaces = ValFRSVariables::DefaultCaptureFaces;
    this->maxCaptureFaces = ValFRSVariables::MaxCaptureFaces;
    this->minFaceLength = ValFRSVariables::MinFaceLength;
    this->maxPitchAngle = ValFRSVariables::MaxPitchAngle;
    this->maxYawAngle = ValFRSVariables::MaxYawAngle;
    this->rois = std::unique_ptr<std::vector<VPolygon>>();
}

void ValFaceCapture::Impl::Detect_Info::Detect_Config_Info(const Napi::Object& config) {
    Detect_Config_Info();

    auto rois = std::unique_ptr<std::vector<VPolygon>>();

    auto oconfidenceScore = config.Get("confidenceScore").As<Napi::Number>();
    if (!oconfidenceScore.IsUndefined()) confidenceScore = oconfidenceScore;

    auto omaxCaptureFaces = config.Get("maxCaptureFaces").As<Napi::Number>();
    if (!omaxCaptureFaces.IsUndefined()) captureFaces = std::min(static_cast<int>(omaxCaptureFaces), maxCaptureFaces);

    auto ominFaceLength = config.Get("minFaceLength").As<Napi::Number>();
    if (!ominFaceLength.IsUndefined()) minFaceLength = ominFaceLength;

    auto omaxYawAngle = config.Get("maxYawAngle").As<Napi::Number>();
    if (!omaxYawAngle.IsUndefined()) maxYawAngle = omaxYawAngle;

    auto omaxPitchAngle = config.Get("maxPitchAngle").As<Napi::Number>();
    if (!omaxPitchAngle.IsUndefined()) maxPitchAngle = omaxPitchAngle;

    /// rois
    auto oRois = config.Get("rois").As<Napi::Array>();
    if (!oRois.IsUndefined()) {
        std::vector<VPolygon> vRois;
        auto roilen = oRois.Length();
        vRois.reserve(roilen);
        /// take all ROIs array
        for (size_t i=0; i<roilen; ++i) {
            auto oRoiArray = oRois.Get(i).As<Napi::Array>();

            VPolygon poly;
            poly.clear();
            for (size_t j=0; j<oRoiArray.Length(); ++j) {
                auto oRoiUnit = oRoiArray.Get(j).As<Napi::Object>();
                poly.add(cv::Point( oRoiUnit.Get("x").As<Napi::Number>(), oRoiUnit.Get("y").As<Napi::Number>() ));
            }
            vRois.emplace_back(std::move(poly));
        }
        rois = std::make_unique<std::vector<VPolygon>>(std::move(vRois));
    }
    this->rois = std::move(rois);
}

ValFaceCapture::Impl::Detect_Info::Detect_Info() :
    dataptr(nullptr),
    callback(nullptr) {
        Detect_Config_Info();
    }

ValFaceCapture::Impl::Detect_Info::Detect_Info(const Napi::Object& o) :
    dataptr(nullptr),
    callback(nullptr) {
        Detect_Config_Info(o);
    }

ValFaceCapture::Impl::Detect_Info::Detect_Info(const Napi::CallbackInfo& info) :
    dataptr(nullptr),
    callback(nullptr) {
    /// When Detect,
    /// 1) If buffer, convert to cv::Mat
    /// 2) If AVFrame, keep it as is
    // std::unique_ptr<ValSharedCapturedFaceData> dataptr = ValSharedCapturedFaceData::FromInputOfFaceCaptureDetect(info);

    size_t argc = info.Length();
    auto bufferOrFrame = info[0];
    this->dataptr = FromBufferOrFrame(bufferOrFrame);
    /// config at 1, callback at 2
    this->callback = std::make_unique<AsyncCaller::AsyncCallerDelegate>(AsyncCaller::SharedInstance()->Wrap(info[argc == 2 ? 1 : 2].As<Napi::Function>()));

    Detect_Config_Info(argc == 3 ? info[1].As<Napi::Object>() : Napi::Object::New(info.Env()));
}