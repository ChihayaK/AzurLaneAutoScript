import socket
import struct
import threading
from functools import wraps
from typing import Optional, Tuple

import cv2
import numpy as np

from module.base.decorator import cached_property
from module.base.decorator import del_cached_property
from module.base.timer import Timer
from module.device.connection import Connection
from module.device.method.minitouch import insert_swipe, random_rectangle_point
from module.device.method.utils import RETRY_TRIES
from module.device.method.utils import retry_sleep
from module.exception import EmulatorNotRunningError, RequestHumanTakeover
from module.logger import logger

## TODO: move to module.device.platform.maatool_platform
class MaaToolsPlatform(Connection):
    """
    MaaTools specific platform integration that skips ADB.
    """
    def detect_device(self):
        if not self.is_maatools_platform:
            super().detect_device()
            return

        if not self.config.MaaTools_Enable:
            logger.error("MaaTools not enabled in config")
            raise RequestHumanTakeover

        try:
            self.maatools
            logger.info(f"Connected to MaaTools server at {self.config.MaaTools_Host}:{self.config.MaaTools_Port}")
        except MaaToolsError as e:
            logger.error(f"Failed to connect to MaaTools: {e}")
            raise EmulatorNotRunningError

    @property
    def is_maatools_platform(self) -> bool:
        return (
            self.config.Emulator_ScreenshotMethod == 'MaaTools'
            or self.config.Emulator_ControlMethod == 'MaaTools'
        )

    def adb_connect(self):
        if self.is_maatools_platform:
            logger.info("Using MaaTools, skipping ADB connection")
            return True
        return super().adb_connect()

    def list_device(self):
        if self.is_maatools_platform:
            devices = []
            try:
                self.maatools
                devices.append({
                    'serial': f"{self.config.MaaTools_Host}:{self.config.MaaTools_Port}",
                    'status': 'device'
                })
            except MaaToolsError:
                pass
            return devices
        return super().list_device()

    def adb_shell(self, cmd, stream=False, recvall=True, timeout=10, rstrip=True):
        """Handle common ADB shell commands without actually using ADB."""
        if not self.is_maatools_platform:
            return super().adb_shell(cmd, stream, recvall, timeout, rstrip)

        if isinstance(cmd, list):
            cmd = ' '.join(cmd)

        if cmd == 'dumpsys display':
            return "DisplayViewport{valid=true, orientation=0, deviceWidth=1280, deviceHeight=720}"
        elif cmd == 'wm size':
            return "Physical size: 1280x720"
        elif cmd == 'settings get system screen_off_timeout':
            return "2147483647"
        else:
            logger.info(f"Using MaaTools, ADB command ignored: {cmd}")
            return ""

    def app_start(self):
        if self.is_maatools_platform:
            logger.info("Using MaaTools, app start/stop not needed")
            return
        super().app_start()

    def app_stop(self):
        if self.is_maatools_platform:
            logger.info("Using MaaTools, app start/stop not needed")
            return
        super().app_stop()

    @cached_property
    def orientation(self) -> int:
        """Get current orientation from MaaTools."""
        return 0  # MaaTools assumes fixed orientation

    def get_orientation(self):
        """Override orientation detection for MaaTools."""
        if self.is_maatools_platform:
            self._orientation = 0
            logger.attr('Device Orientation', f'0 (Normal)')
            return 0
        return super().get_orientation()

    def touch_init(self):
        """Init touch input mode."""
        if self.is_maatools_platform:
            logger.hr('MaaTools Touch Init')
            return True
        return super().touch_init()

    def click(self, button, count=1):
        """
        Args:
            button: Button instance.
            count (int): Default to 1.
        """
        if self.is_maatools_platform:
            for _ in range(count):
                x, y = random_rectangle_point(button.button) if isinstance(button.button, tuple) else button.button
                self.click_maatools(x, y)
                self.sleep(0.1)
            return
        return super().click(button, count)

    def long_click(self, button, duration=1.0):
        """
        Args:
            button: Button instance.
            duration (float): Duration of long click, in seconds.
        """
        if self.is_maatools_platform:
            x, y = random_rectangle_point(button.button) if isinstance(button.button, tuple) else button.button
            self.long_click_maatools(x, y, duration)
            return
        return super().long_click(button, duration)

    def swipe(self, p1, p2, duration=0.2):
        """
        Args:
            p1 (tuple): Start point, (x, y).
            p2 (tuple): End point, (x, y).
            duration (float): Duration of swipe motion.
        """
        if self.is_maatools_platform:
            self.swipe_maatools(p1, p2, duration)
            return
        return super().swipe(p1, p2, duration)

    def drag(self, p1, p2, segments=1, shake=(0, 15, 0, 0), point_random=(-10, -10, 10, 10), shake_random=(-1, -1, 1, 1),
            swipe_duration=0.25, shake_duration=0.1):
        """
        Drag and shake.
        Args:
            p1 (tuple): Start point, (x, y).
            p2 (tuple): End point, (x, y).
            segments (int):
            shake (tuple): Shake showing as up, right, down, left.
            point_random: Random shift after arriving end point.
            shake_random: Random shift after shake.
            swipe_duration: Duration of the swipe motion.
            shake_duration: Duration of the shake motion.
        """
        if self.is_maatools_platform:
            self.drag_maatools(p1, p2, point_random=point_random)
            return
        return super().drag(p1, p2, segments, shake, point_random, shake_random, swipe_duration, shake_duration)

    def get_window_size(self) -> tuple:
        """Get window size."""
        if self.is_maatools_platform:
            return self.max_x, self.max_y
        return super().get_window_size()

    @property
    def maatouch_builder(self):
        """Override maatouch_builder to prevent MaaTouch initialization."""
        if self.is_maatools_platform:
            logger.info('Using MaaTools, MaaTouch builder disabled')
            return None
        return super().maatouch_builder

    @property
    def minitouch_builder(self):
        """Override minitouch_builder to prevent minitouch initialization."""
        if self.is_maatools_platform:
            logger.info('Using MaaTools, minitouch builder disabled')
            return None
        return super().minitouch_builder

    @cached_property
    def touch_method(self):
        """
        Returns:
            str: 'minitouch', 'maatouch', 'maatools', or 'adb'
        """
        if self.is_maatools_platform:
            logger.info('Using MaaTools touch method')
            return 'maatools'
        return super().touch_method

    def handle_control(self, button):
        """Handle click control."""
        if self.is_maatools_platform:
            self.click_maatools(button.x, button.y)
            return
        return super().handle_control(button)


class MaaToolsError(Exception):
    """Base exception for MaaTools-related errors."""
    pass


class MaaToolsNotInstalledError(MaaToolsError):
    """Raised when MaaTools is not installed or incompatible."""
    pass


class Command:
    """Command wrapper for MaaTools protocol."""
    def __init__(
            self,
            operation: str,
            phase: int = 0,
            x: int = 0,
            y: int = 0,
            ms: int = 10,
            pressure: int = 100,
            text: str = ''
    ):
        self.operation = operation
        self.phase = phase
        self.x = x
        self.y = y
        self.ms = ms
        self.pressure = pressure
        self.text = text

    def to_bytes(self) -> bytes:
        """Convert command to MaaTools binary format."""
        if self.operation == 'touch':
            # Pack touch command: phase, x, y
            payload = struct.pack('>BHH', self.phase, self.x, self.y)
            payload_length = 4 + len(payload)  # 4 bytes for magic + payload
            return struct.pack('>H', payload_length) + b'TUCH' + payload
        elif self.operation == 'wait':
            # Just return empty bytes for wait - handled by sleep in send
            return b''
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


class CommandBuilder:
    """Builds sequences of MaaTools protocol commands."""
    DEFAULT_DELAY = 0.05

    def __init__(self, device):
        self.device = device
        self.commands = []
        self.delay = 0

    def clear(self):
        """Clear all pending commands."""
        self.commands = []
        self.delay = 0
        return self

    def down(self, x: int, y: int) -> 'CommandBuilder':
        """Add a touch down command."""
        x, y = self.convert(x, y)
        self.commands.append(Command('touch', phase=0, x=x, y=y))
        return self

    def move(self, x: int, y: int) -> 'CommandBuilder':
        """Add a touch move command."""
        x, y = self.convert(x, y)
        self.commands.append(Command('touch', phase=1, x=x, y=y))
        return self

    def up(self) -> 'CommandBuilder':
        """Add a touch up command."""
        self.commands.append(Command('touch', phase=3))
        return self

    def wait(self, ms: int) -> 'CommandBuilder':
        """Add a wait command."""
        self.commands.append(Command('wait', ms=ms))
        self.delay += ms
        return self

    def convert(self, x, y):
        """Convert coordinates based on device orientation."""
        if not self.device.config.DEVICE_OVER_HTTP:
            # Maximum X and Y coordinates may, but usually do not, match the display size.
            x, y = int(x / 1280 * self.device.maatools.max_x), int(y / 720 * self.device.maatools.max_y)
        return x, y

    def to_commands(self) -> bytes:
        """Convert all commands to binary format."""
        return b''.join([cmd.to_bytes() for cmd in self.commands if cmd.operation != 'wait'])


def retry(func):
    """MaaTools specific retry decorator."""
    @wraps(func)
    def retry_wrapper(self, *args, **kwargs):
        """
        Args:
            self (MaaToolsImpl):
        """
        init = None
        for _ in range(RETRY_TRIES):
            try:
                if callable(init):
                    retry_sleep(_)
                    init()
                return func(self, *args, **kwargs)
            except MaaToolsNotInstalledError as e:
                logger.error(e)
                break
            except MaaToolsError as e:
                logger.error(e)

                def init():
                    self.maatools_connect()
            except Exception as e:
                logger.exception(e)

                def init():
                    pass

        logger.critical(f'Retry {func.__name__}() failed')
        raise RequestHumanTakeover

    return retry_wrapper


class MaaToolsImpl:
    """Core MaaTools implementation."""

    def __init__(self, host='localhost', port=17233):
        """
        Args:
            host (str): Server host
            port (int): Server port
            device: Device instance
        """
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.width = 1280
        self.height = 720
        self.lock = threading.Lock()
        self.max_x = 1280
        self.max_y = 720
        self.connected = False

    @retry
    def maatools_connect(self):
        """Connect and handshake with server."""
        if self.connected:
            return

        logger.hr('MaaTools Connect')
        if self.sock is not None:
            self.sock.close()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host, self.port))
            # Handshake
            self.sock.sendall(b'MAA\x00')  # Magic bytes
            resp = self.sock.recv(4)
            if resp != b'OKAY':
                raise MaaToolsError(f'Handshake failed: {resp}')

            # Get window size
            self.width, self.height = self._get_window_size()
            logger.attr('Connect', f'{self.host}:{self.port}')
            logger.attr('Window', f'{self.width}x{self.height}')
            self.connected = True
        except Exception as e:
            self.sock = None
            self.connected = False
            raise MaaToolsError(f"Connection failed: {e}")

    def maatools_disconnect(self):
        """Close connection."""
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
            finally:
                self.sock = None

    def _get_window_size(self) -> Tuple[int, int]:
        """Get window dimensions."""
        message = struct.pack('>H', 4) + b'SIZE'
        self.sock.sendall(message)
        size_data = self.sock.recv(4)
        width, height = struct.unpack('>HH', size_data)
        self.max_x = width
        self.max_y = height
        return width, height


class MaaTools(MaaToolsPlatform):
    """MaaTools platform integration."""
    _screenshot_interval = Timer(0.1)
    _maatools: Optional[MaaToolsImpl] = None

    @property
    def is_maatools_platform(self) -> bool:
        return (
                self.config.Emulator_ScreenshotMethod == 'MaaTools'
                or self.config.Emulator_ControlMethod == 'MaaTools'
        )

    @cached_property
    def maatools(self) -> MaaToolsImpl:
        """
        Returns:
            MaaToolsImpl:
        """
        if not self.config.MaaTools_Enable:
            raise MaaToolsError("MaaTools not enabled in config")

        self._maatools = MaaToolsImpl(
            host=self.config.MaaTools_Host,
            port=self.config.MaaTools_Port,
        )
        self._maatools.maatools_connect()
        return self._maatools

    def maatools_release(self):
        """Release MaaTools resources."""
        if self._maatools is not None:
            self._maatools.maatools_disconnect()
        del_cached_property(self, 'maatools')
        logger.info('MaaTools released')

    @cached_property
    def maatools_builder(self):
        """Command builder instance."""
        return CommandBuilder(self)

    def screenshot_maatools(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Screenshot in BGR format
        """
        if not self.is_maatools_platform:
            raise MaaToolsError("Screenshot method not set to MaaTools")

        timeout = max(self._screenshot_interval.limit - 0.01, 0.1)

        # Request screenshot
        message = struct.pack('>H', 4) + b'SCRN'
        self.maatools.sock.sendall(message)

        # Get image size
        size_data = self.maatools.sock.recv(4)
        image_size = struct.unpack('>I', size_data)[0]

        # Get image data
        image_data = self._recv_exact(image_size)

        # Expected size checks
        channels = image_size // (self.maatools.height * self.maatools.width)
        try:
            if channels == 4:
                # RGBA format
                image = np.frombuffer(image_data, dtype=np.uint8)
                image = image.reshape((self.maatools.height, self.maatools.width, 4))
                # Convert RGBA to BGR using cv2
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif channels == 3:
                # RGB format
                image = np.frombuffer(image_data, dtype=np.uint8)
                image = image.reshape((self.maatools.height, self.maatools.width, 3))
                # Convert RGB to BGR using cv2
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                raise MaaToolsError(f"Unexpected image format with {channels} channels")

            # Make sure we return a contiguous array with correct type
            image = np.ascontiguousarray(image, dtype=np.uint8)

            # Validate output format
            if image.shape != (self.maatools.height, self.maatools.width, 3):
                raise MaaToolsError(
                    f"Invalid image shape: {image.shape}, "
                    f"expected ({self.maatools.height}, {self.maatools.width}, 3)"
                )

            # Log first successful screenshot
            if not hasattr(self, '_first_maatools_screenshot'):
                logger.info(
                    f'First MaaTools screenshot: '
                    f'shape={image.shape}, dtype={image.dtype}, channels={channels}'
                )
                self._first_maatools_screenshot = True

            return image

        except Exception as e:
            logger.error(f'Screenshot error: {e}')
            logger.error(f'Image size: {image_size}, expected channels: {channels}')
            logger.error(f'Expected dimensions: {self.maatools.height}x{self.maatools.width}')
            raise

    def _recv_exact(self, size: int) -> bytes:
        """
        Receive exact number of bytes.

        Args:
            size (int): Number of bytes to receive

        Returns:
            bytes: Received data
        """
        data = b''
        while len(data) < size:
            chunk = self.maatools.sock.recv(size - len(data))
            if not chunk:
                raise MaaToolsError("Connection closed while receiving data")
            data += chunk
        return data

    def click_maatools(self, x, y):
        """Perform click operation."""
        if not self.is_maatools_platform:
            raise MaaToolsError("Control method not set to MaaTools")

        builder = self.maatools_builder
        builder.clear().down(x, y).wait(10)
        self.send_commands(builder)

        builder.clear().up()
        self.send_commands(builder)

    def long_click_maatools(self, x, y, duration=1.0):
        """Perform long click operation."""
        if not self.is_maatools_platform:
            raise MaaToolsError("Control method not set to MaaTools")

        duration_ms = int(duration * 1000)
        builder = self.maatools_builder
        builder.clear().down(x, y).wait(duration_ms)
        self.send_commands(builder)

        builder.clear().up()
        self.send_commands(builder)

    def swipe_maatools(self, p1, p2):
        """Perform swipe operation."""
        if not self.is_maatools_platform:
            raise MaaToolsError("Control method not set to MaaTools")

        points = insert_swipe(p0=p1, p3=p2)
        builder = self.maatools_builder

        builder.clear().down(*points[0]).wait(10)
        self.send_commands(builder)

        for point in points[1:]:
            builder.clear().move(*point).wait(10)
            self.send_commands(builder)

        builder.clear().up()
        self.send_commands(builder)

    def drag_maatools(self, p1, p2, point_random=(-10, -10, 10, 10)):
        """Perform drag operation."""
        if not self.is_maatools_platform:
            raise MaaToolsError("Control method not set to MaaTools")

        p1 = np.array(p1) - random_rectangle_point(point_random)
        p2 = np.array(p2) - random_rectangle_point(point_random)
        points = insert_swipe(p0=tuple(p1), p3=tuple(p2), speed=20)

        builder = self.maatools_builder
        builder.clear().down(*points[0]).wait(10)
        self.send_commands(builder)

        for point in points[1:]:
            builder.clear().move(*point).wait(10)
            self.send_commands(builder)

        builder.clear().move(*p2).wait(140)
        self.send_commands(builder)
        builder.clear().move(*p2).wait(140)
        self.send_commands(builder)

        builder.clear().up()
        self.send_commands(builder)

    def send_commands(self, builder: CommandBuilder):
        """Send commands and handle delays."""
        command_bytes = builder.to_commands()
        with self.maatools.lock:
            self.maatools.sock.sendall(command_bytes)
            if builder.delay > 0:
                self.sleep(builder.delay / 1000)
            self.sleep(builder.DEFAULT_DELAY)
        builder.clear()

    def detect_platform(self):
        """Check if this is a MaaTools device."""
        if not self.is_maatools_platform:
            return False

        try:
            self.maatools
            return True
        except MaaToolsError:
            return False

    def app_is_running(self) -> bool:
        """Check if emulator is running."""
        return self.detect_platform()

    def release_resource(self):
        self.maatools_release()
        super().release_resource()