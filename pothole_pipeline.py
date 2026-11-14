import argparse
import cv2
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Union

import torch
from ultralytics import YOLO


CONF_THRESHOLD = 0.40
DEFAULT_IMG_SIZE = 640

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s"
)

logger = logging.getLogger("PotholeSystem")


class GpsSender:
    """
    GPS 담당 모듈과 연결하기 위한 인터페이스.

    현재는 JSONL 파일에 기록.
    이후 Socket / MQTT / HTTP / ZeroMQ 등으로
    send() 내부만 교체
    """

    def __init__(self, log_path: str = "logs/detections.jsonl"):

        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

    def send(self, payload: Dict[str, Any]):

       
        with self.log_path.open(
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(
                    payload,
                    ensure_ascii=False
                )
                + "\n"
            )

        # ------------------------------------------
        # 실제 GPS 통신 구현 시 아래를 교체
        # ------------------------------------------

        # 예:
        #
        # socket.send(...)
        #
        # mqtt_client.publish(...)
        #
        # requests.post(...)
        #
        # zmq_socket.send_json(...)



class PotholeDetectorPipeline:

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.40,
        imgsz: int = 640,
        sender: GpsSender = None
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"모델 파일을 찾을 수 없습니다:\n"
                f"{self.model_path}"
            )

        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

        # GPU 자동 선택
        if torch.cuda.is_available():

            self.device = 0

            logger.info(
                f"GPU 사용: "
                f"{torch.cuda.get_device_name(0)}"
            )

        else:

            self.device = "cpu"

            logger.info("CPU 사용")

        self.model = self._load_model()

        self.sender = sender or GpsSender()



    def _load_model(self):

        logger.info(
            f"AI 모델 로드 중: {self.model_path}"
        )

        model = YOLO(
            str(self.model_path)
        )

        logger.info(
            f"모델 클래스: {model.names}"
        )

        return model


    def run_inference(
        self,
        frame,
        frame_id: int,
        timestamp: float
    ) -> Dict[str, Any]:

        """
        입력
        ----
        frame
            카메라 프레임

        frame_id
            현재 프레임 번호

        timestamp
            프레임 수신 시각

        반환
        ----
        JSON으로 변환 가능한 Python Dict
        """

        start_time = time.perf_counter()

        
        results = self.model.predict(

            source=frame,

            conf=self.conf_threshold,

            imgsz=self.imgsz,

            device=self.device,

            verbose=False
        )

        result = results[0]

        potholes: List[Dict[str, Any]] = []



        if result.boxes is not None:

            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )

                # 혹시 모를 추가 필터
                if confidence < self.conf_threshold:
                    continue


                class_id = int(
                    box.cls[0]
                )


                class_name = self.model.names[
                    class_id
                ]


                # pothole 클래스만 사용
                if class_name.lower() != "pothole":
                    continue


                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )


                pothole = {

                    # 기존 GPS팀 협의 규격 유지
                    "timestamp": timestamp,

                    "confidence": round(
                        confidence,
                        4
                    ),

                    "bbox": {

                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                }

                potholes.append(
                    pothole
                )


        inference_ms = (
            time.perf_counter()
            - start_time
        ) * 1000


    
        payload = {

            # 나중에 규격이 바뀔 경우를 위한 버전
            "schema_version": "1.0",

            # 프레임 추적
            "frame_id": frame_id,

            # 프레임 수신 시간
            "timestamp": timestamp,

            # UTC 확인용
            "timestamp_iso": (
                datetime
                .fromtimestamp(
                    timestamp,
                    tz=timezone.utc
                )
                .isoformat()
            ),

            # 검출 개수
            "total_count": len(
                potholes
            ),

            # 실제 검출 결과
            "potholes": potholes,

            # 성능 확인용
            "inference_ms": round(
                inference_ms,
                2
            )
        }

        return payload


    # ======================================================
    # GPS 모듈 전달
    # ======================================================

    def send_to_gps_module(
        self,
        payload: Dict[str, Any]
    ):

        # 현재는 포트홀이 검출된 경우에만 전달
        if payload["total_count"] == 0:
            return

        self.sender.send(
            payload
        )


    # ======================================================
    # Bounding Box 시각화
    # ======================================================

    def draw_bounding_boxes(
        self,
        frame,
        payload: Dict[str, Any]
    ):

        for det in payload["potholes"]:

            box = det["bbox"]

            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

            confidence = det["confidence"]


            # 바운딩 박스
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Confidence
            label = (
                f"Pothole "
                f"{confidence:.2f}"
            )


            cv2.putText(

                frame,

                label,

                (
                    x1,
                    max(y1 - 10, 20)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 0),

                2
            )

        return frame


def run_system(
    model_path: str,
    source: Union[int, str],
    conf: float,
    imgsz: int,
    output_path: str = None,
    show: bool = True
):

    sender = GpsSender(
        "logs/detections.jsonl"
    )


    pipeline = PotholeDetectorPipeline(

        model_path=model_path,

        conf_threshold=conf,

        imgsz=imgsz,

        sender=sender
    )


    # ======================================================
    # 카메라 / 영상 열기
    # ======================================================

    cap = cv2.VideoCapture(
        source
    )


    if not cap.isOpened():

        raise RuntimeError(
            f"영상 또는 카메라를 "
            f"열 수 없습니다: {source}"
        )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    if fps <= 0:
        fps = 30


    # ======================================================
    # 결과 영상 저장
    # ======================================================

    writer = None


    if output_path:

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        writer = cv2.VideoWriter(

            str(output_path),

            cv2.VideoWriter_fourcc(
                *"mp4v"
            ),

            fps,

            (
                width,
                height
            )
        )


    logger.info(
        "포트홀 검출 시스템 시작"
    )


    frame_id = 0


    # ======================================================
    # 메인 Loop
    # ======================================================

    try:

        while cap.isOpened():

            ret, frame = cap.read()


            if not ret:

                logger.info(
                    "영상 입력 종료"
                )

                break


            # ----------------------------------------------
            # 가장 중요한 부분:
            # 프레임을 받은 즉시 Timestamp 기록
            # ----------------------------------------------

            frame_timestamp = time.time()


            # ----------------------------------------------
            # 1. AI 추론
            # ----------------------------------------------

            payload = pipeline.run_inference(

                frame=frame,

                frame_id=frame_id,

                timestamp=frame_timestamp
            )


            # ----------------------------------------------
            # 2. GPS 모듈 전달
            # ----------------------------------------------

            pipeline.send_to_gps_module(
                payload
            )


            # ----------------------------------------------
            # 3. Bounding Box
            # ----------------------------------------------

            visualized = (
                pipeline
                .draw_bounding_boxes(
                    frame,
                    payload
                )
            )


            # ----------------------------------------------
            # 시스템 상태 표시
            # ----------------------------------------------

            status_text = (

                f"Frame: {frame_id} | "

                f"Potholes: "
                f"{payload['total_count']} | "

                f"Inference: "
                f"{payload['inference_ms']:.1f}ms"
            )


            cv2.putText(

                visualized,

                status_text,

                (20, 35),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2
            )


            # ----------------------------------------------
            # 결과 영상 저장
            # ----------------------------------------------

            if writer is not None:

                writer.write(
                    visualized
                )


            # ----------------------------------------------
            # 화면 표시
            # ----------------------------------------------

            if show:

                cv2.imshow(

                    "Pothole Detection System",

                    visualized
                )


                if (
                    cv2.waitKey(1)
                    & 0xFF
                    == ord("q")
                ):

                    break


            # ----------------------------------------------
            # 디버깅 로그
            # ----------------------------------------------

            if payload["total_count"] > 0:

                logger.info(

                    f"Frame {frame_id}: "
                    f"{payload['total_count']} "
                    f"pothole detected"
                )


            frame_id += 1


    except KeyboardInterrupt:

        logger.info(
            "사용자가 시스템을 종료했습니다."
        )


    finally:

        cap.release()


        if writer is not None:
            writer.release()


        cv2.destroyAllWindows()


        logger.info(
            "포트홀 검출 시스템 종료"
        )



def main():

    parser = argparse.ArgumentParser(
        description=
        "Pothole Detection Pipeline"
    )


    parser.add_argument(
        "--model",
        default="best6.pt",
        help="YOLO 모델 경로"
    )


    parser.add_argument(
        "--source",
        default="0",
        help=
        "카메라 번호 또는 영상 파일 경로"
    )


    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD,
        help="Confidence threshold"
    )


    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="YOLO inference image size"
    )


    parser.add_argument(
        "--output",
        default=None,
        help="결과 영상 저장 경로"
    )


    parser.add_argument(
        "--no-show",
        action="store_true",
        help="화면 표시 비활성화"
    )


    args = parser.parse_args()


    # source가 "0"이면 웹캠 번호 0
    if args.source.isdigit():

        source = int(
            args.source
        )

    else:

        source = args.source


    run_system(

        model_path=args.model,

        source=source,

        conf=args.conf,

        imgsz=args.imgsz,

        output_path=args.output,

        show=not args.no_show
    )



if __name__ == "__main__":
    main()