"""RKNNLite-only model executor for RK3588 deployment."""


class RKNN_model_container:
    def __init__(self, model_path, target=None, device_id=None, core_mask="NPU_CORE_1") -> None:
        _ = target
        _ = device_id
        from rknnlite.api import RKNNLite

        rknn = RKNNLite()
        ret = rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError("Load RKNN model failed: {}".format(ret))
        mask_value = getattr(RKNNLite, str(core_mask), None)
        if mask_value is None:
            rknn.release()
            raise ValueError("Unsupported RKNNLite core mask: {}".format(core_mask))
        print("--> Init RKNNLite runtime environment, core_mask={}".format(core_mask))
        ret = rknn.init_runtime(core_mask=mask_value)
        if ret != 0:
            rknn.release()
            raise RuntimeError("Init runtime environment failed: {}".format(ret))
        self.rknn = rknn

    def run(self, inputs):
        if self.rknn is None:
            raise RuntimeError("RKNN model has been released")
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        return self.rknn.inference(inputs=inputs)

    def release(self):
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
