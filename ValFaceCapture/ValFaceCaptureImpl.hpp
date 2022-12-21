#ifndef _ValFaceCaptureImpl_h
#define _ValFaceCaptureImpl_h

#include <napi.h>
#include <memory>

/// forward declaration - opencv
#include "fwd/opencv.h"

/// forward declaration - ffmpeg
#include "fwd/ffmpeg.h"

/// forward declaration - spdlog
#include "fwd/spdlog_full.h"

#include "val-frs-global.hpp"
#include "val-frs-variables.hpp"
#include "polygon-intersection.hpp"

#include "impl-hpp-include.h"

/// For ValSharedCapturedFaceData
#include "../ValSharedObjects/ValSharedCapturedFace/ValSharedCapturedFaceData.hpp"
#include "facedetectcnn.h"

class ValFaceCapture::Impl {
    public:
        void SetLogLevel(const spdlog::level::level_enum& logLevel);
    public:
        class Ctor_Info {
            public:
                Ctor_Info();
                Ctor_Info(const Napi::CallbackInfo& info);
        };
        Impl(const Ctor_Info& info);
        Impl();

    public:
        class Detect_Info {
            public:
                std::shared_ptr<ValSharedCapturedFaceData> dataptr;
                std::shared_ptr<AsyncCaller::AsyncCallerDelegate> callback;
                float confidenceScore;
                int captureFaces;
                int maxCaptureFaces;
                int minFaceLength;
                float maxPitchAngle;
                float maxYawAngle;
                std::shared_ptr<std::vector<VPolygon>> rois;
            public:
                Detect_Info();
                Detect_Info(const Napi::CallbackInfo& info);
                Detect_Info(const Napi::Object& o);
            private:
                void Detect_Config_Info();
                void Detect_Config_Info(const Napi::Object& o);
            private:
                std::unique_ptr<ValSharedCapturedFaceData> FromBufferOrFrame(const Napi::Value& bufferOrFrame);
        };
        std::vector<std::unique_ptr<ValSharedCapturedFaceData>> Detect(const Detect_Info& info);

    private:
        std::vector<std::unique_ptr<ValSharedCapturedFaceData>> detect(const std::shared_ptr<cv::Mat>& data,
                int maxCaptureFaces = ValFRSVariables::MaxCaptureFaces, float confidenceScore = ValFRSVariables::ConfidenceScore,
                int minFaceLength = ValFRSVariables::MinFaceLength,
                float maxYawAngle = ValFRSVariables::MaxYawAngle, float maxPitchAngle = ValFRSVariables::MaxPitchAngle
            );
        std::vector<std::unique_ptr<ValSharedCapturedFaceData>> detect(const std::shared_ptr<AVFrame>& data,
                int maxCaptureFaces = ValFRSVariables::MaxCaptureFaces, float confidenceScore = ValFRSVariables::ConfidenceScore,
                int minFaceLength = ValFRSVariables::MinFaceLength,
                float maxYawAngle = ValFRSVariables::MaxYawAngle, float maxPitchAngle = ValFRSVariables::MaxPitchAngle
            );
        std::vector<std::unique_ptr<ValSharedCapturedFaceData>> detect(const ValSharedCapturedFaceData& data,
                int maxCaptureFaces = ValFRSVariables::MaxCaptureFaces, float confidenceScore = ValFRSVariables::ConfidenceScore,
                int minFaceLength = ValFRSVariables::MinFaceLength,
                float maxYawAngle = ValFRSVariables::MaxYawAngle, float maxPitchAngle = ValFRSVariables::MaxPitchAngle
            );

    private:
        /// model
        std::shared_ptr<ThreadSafeModel>& model;

    private:
        /// logger
        constexpr static auto logTitle = "ValFaceCapture";
        static std::shared_ptr<spdlog::logger> sharedLogger;
        std::shared_ptr<spdlog::logger> logger;
};

#endif