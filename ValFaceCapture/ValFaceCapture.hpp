#ifndef _ValFaceCapture_h
#define _ValFaceCapture_h

#include <napi.h>

#include "impl-hpp-include.h"

class ValFaceCapture : public Napi::ObjectWrap<ValFaceCapture> {
    public:
        ValFaceCapture(const Napi::CallbackInfo &info);
        virtual void Finalize(const Napi::Env& env);
    /// custom
    public:
        Napi::Value Detect(const Napi::CallbackInfo& info);

    private:
        /// Log
        void SetLogLevel(const Napi::CallbackInfo& info, const Napi::Value& value);

    /// nodejs internal use
    public:
        static Napi::FunctionReference constructor;
    public:
        static Napi::Object Init(Napi::Env& env, Napi::Object& exports);

    private:
        /// thread pool
        std::shared_ptr<ValThreadPool>& pool;

    /// impl
    public:
        class Impl;
    private:
        std::unique_ptr<Impl> pImpl;
};

#endif