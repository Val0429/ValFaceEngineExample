#include "ValFaceCapture.hpp"
#include "ValFaceCaptureImpl.hpp"

#include "../ValSharedObjects/ValSharedCapturedFace/ValSharedCapturedFace.hpp"

#include "resource-remover.hpp"

#include "impl-cpp-include.h"


ValFaceCapture::ValFaceCapture(const Napi::CallbackInfo& info) :
    Napi::ObjectWrap<ValFaceCapture>(info),
    pool(ValFRSGlobal::SharedThreadPool()),
    pImpl( std::make_unique<Impl>(Impl::Ctor_Info(info)) ) {}

void ValFaceCapture::Finalize(const Napi::Env& env) {
    pImpl.reset();
}

Napi::Value ValFaceCapture::Detect(const Napi::CallbackInfo& info) {
    auto r1 = ResourceRemover::make_shared([this] { this->Ref(); }, [this] { this->Unref(); });
    auto vinfo = Impl::Detect_Info(info);

    pool->submit([this,
        vinfo = std::move(vinfo), r1 = std::move(r1)
    ]() mutable {
        auto finaldatas = pImpl->Detect(vinfo);
        std::promise<void> ready;
        vinfo.callback->Call([this, &ready, datas = std::move(finaldatas), r1 = std::move(r1)](Napi::Env&& env, Napi::Function&& callback) mutable {
            auto size = datas.size();
            auto rtn = Napi::Array::New(env, size);
            for (unsigned int i=0; i<size; ++i) {
                auto& data = datas[i];
                auto pdata = data.get();
                data.release();
                auto exone = Napi::External<ValSharedCapturedFaceData>::New(env, pdata);
                auto shared = ValSharedCapturedFace::constructor.New({ std::move(exone) });
                rtn.Set(i, std::move(shared));
            }
            callback.Call({ env.Undefined(), std::move(rtn) });
            ready.set_value();
        });
        ready.get_future().wait();
    });

    return info.Env().Undefined();
}

void ValFaceCapture::SetLogLevel(const Napi::CallbackInfo& info, const Napi::Value& value) {
    pImpl->SetLogLevel(GetLogLevel(value.As<Napi::String>()));
}

/// private property initial
Napi::FunctionReference ValFaceCapture::constructor;

/// private Initialization
Napi::Object ValFaceCapture::Init(Napi::Env& env, Napi::Object& exports) {
    Napi::Function func = DefineClass(env, "ValFaceCapture", {
        InstanceMethod<&ValFaceCapture::Detect>("detectWithCallback"),

        InstanceAccessor("logLevel", NULL, &ValFaceCapture::SetLogLevel)
    });

    constructor = Napi::Persistent(func);
    constructor.SuppressDestruct();

    exports.Set("ValFaceCapture", func);
    return exports;
}
