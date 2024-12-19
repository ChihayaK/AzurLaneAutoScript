import socket
import struct
import threading
from functools import wraps
from typing import Optional, Tuple

import cv2
import numpy as np

from module.base.decorator import cached_property, del_cached_property
from module.base.timer import Timer
from module.config.server import set_server
from module.device.method.adb import Adb
from module.device.method.minitouch import insert_swipe, random_rectangle_point
from module.device.method.utils import RETRY_TRIES
from module.device.method.utils import retry_sleep
from module.device.platform.emulator_base import EmulatorInstanceBase
from module.device.platform.platform_base import PlatformBase
from module.exception import EmulatorNotRunningError, RequestHumanTakeover
from module.logger import logger
from module.map.map_grids import SelectedGrids


def retry(func):
    """Retry decorator for MaaTools operations."""
    @wraps(func)
    def retry_wrapper(self, *args, **kwargs):
        init = None
        for _ in range(RETRY_TRIES):
            try:
                if callable(init):
                    retry_sleep(_)
                    init()
                return func(self, *args, **kwargs)
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


class MaaToolsEmulatorInstance(EmulatorInstanceBase):
    """
    A mock emulator instance for MaaTools connections.
    """
    def __init__(self, serial='127.0.0.1:17233', name='MaaTools', path='MaaTools'):
        super().__init__(serial=serial, name=name, path=path)
        self._type = 'MaaTools'  # Store locally to avoid EmulatorBase lookup

    @property
    def type(self) -> str:
        """Override type to avoid EmulatorBase lookup."""
        return self._type

    def __bool__(self):
        """Always valid."""
        return True

    def __str__(self):
        return f'MaaToolsEmulatorInstance(serial="{self.serial}", name="{self.name}", type="{self.type}")'

class MaaToolsPlatform(PlatformBase):
    """
    Platform integration for MaaTools connections.
    Mimics an emulator platform to satisfy system requirements.
    """
    def __init__(self, config):
        """
        Args:
            config (AzurLaneConfig):
        """
        super().__init__(config)

    @cached_property
    def emulator_instance(self) -> MaaToolsEmulatorInstance:
        """Override to return MaaTools instance."""
        return MaaToolsEmulatorInstance(
            serial=f"{self.config.MaaTools_Host}:{self.config.MaaTools_Port}",
            name='MaaTools'
        )

    def detect_platform(self):
        """Override platform detection."""
        return True if self.is_maatools_platform else False

    def iter_running_emulator(self):
        """MaaTools doesn't track running emulators."""
        if self.is_maatools_platform:
            return []
        return super().iter_running_emulator()

    def find_emulator_instance(self, serial: str, name: str = None, path: str = None, emulator: str = None):
        """Override to handle MaaTools instance."""
        if self.is_maatools_platform:
            return MaaToolsEmulatorInstance(
                serial=serial or f"{self.config.MaaTools_Host}:{self.config.MaaTools_Port}",
                name=name or 'MaaTools',
                path=path or '',
                type=emulator or 'MaaTools'
            )
        return super().find_emulator_instance(serial, name, path, emulator)

    def all_emulator_instances(self):
        """Override to include MaaTools instance."""
        if self.is_maatools_platform:
            return [self.emulator_instance]
        return super().all_emulator_instances()

    def method_check(self):
        """Bypass method checks for MaaTools."""
        if self.is_maatools_platform:
            return
        return super().method_check()


class MaaToolsCommand:
    """Command wrapper for MaaTools protocol."""
    def __init__(
            self,
            operation: str,
            phase: int = 0,
            x: Optional[int] = None,
            y: Optional[int] = None,
            ms: int = 10,
            pressure: int = 100
    ):
        self.operation = operation
        self.phase = phase
        self.x = x if x is not None else 0
        self.y = y if y is not None else 0
        self.ms = ms
        self.pressure = pressure

    def to_bytes(self) -> bytes:
        """Convert command to MaaTools binary format."""
        if self.operation == 'touch':
            # Ensure x and y are provided for touch operations
            payload = struct.pack('>BHH', self.phase, self.x, self.y)
            payload_length = 4 + len(payload)
            return struct.pack('>H', payload_length) + b'TUCH' + payload
        elif self.operation == 'wait':
            return b''
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")


class MaaToolsBuilder:
    """Builds sequences of MaaTools protocol commands."""
    DEFAULT_DELAY = 0.1

    def __init__(self, device):
        self.device = device
        self.commands = []
        self.delay = 0

    def clear(self):
        """Clear all pending commands."""
        self.commands = []
        self.delay = 0
        return self

    def down(self, x: int, y: int) -> 'MaaToolsBuilder':
        """Add a touch down command."""
        x, y = self.convert(x, y)
        self.commands.append(MaaToolsCommand('touch', phase=0, x=x, y=y))
        return self

    def move(self, x: int, y: int) -> 'MaaToolsBuilder':
        """Add a touch move command."""
        x, y = self.convert(x, y)
        self.commands.append(MaaToolsCommand('touch', phase=1, x=x, y=y))
        return self

    def up(self, x: int = 0, y: int = 0) -> 'MaaToolsBuilder':
        """Add a touch up command."""
        self.commands.append(MaaToolsCommand('touch', phase=3, x=x, y=y))
        return self

    def wait(self, ms: int) -> 'MaaToolsBuilder':
        """Add a wait command."""
        self.commands.append(MaaToolsCommand('wait', ms=ms))
        self.delay += ms
        return self

    def convert(self, x, y):
        """Convert coordinates based on device orientation."""
        original_x, original_y = x, y
        if not self.device.config.DEVICE_OVER_HTTP:
            x = int(x / 1280 * self.device.maatools.max_x)
            y = int(y / 720 * self.device.maatools.max_y)
            # logger.info(f'Coordinate conversion: ({original_x}, {original_y}) -> ({x}, {y})')
            # logger.info(f'Using scaling factors: {self.device.maatools.max_x}/1280, {self.device.maatools.max_y}/720')
        return x, y

    def to_commands(self) -> bytes:
        """Convert all commands to binary format."""
        return b''.join([cmd.to_bytes() for cmd in self.commands if cmd.operation != 'wait'])


class MaaToolsImpl:
    """Core MaaTools implementation."""

    def __init__(self, host='localhost', port=17233):
        """
        Args:
            host (str): Server host
            port (int): Server port
        """
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
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
            self.sock.sendall(b'MAA\x00')
            resp = self.sock.recv(4)
            if resp != b'OKAY':
                raise MaaToolsError(f'Handshake failed: {resp}')

            # Get window size
            self.max_x, self.max_y = self._get_window_size()
            logger.attr('Connect', f'{self.host}:{self.port}')
            logger.attr('Window', f'{self.max_x}x{self.max_y}')
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
        return width, height


class MaaToolsError(Exception):
    """Base exception for MaaTools-related errors."""
    pass


class MaaTools(MaaToolsPlatform):
    """
    MaaTools platform integration that replaces ADB.
    Inherits from both Adb and MaaToolsPlatform to get both functionalities.
    """

    def __init__(self, config):
        """
        Args:
            config (AzurLaneConfig):
        """
        super().__init__(config)

        # Connect MaaTools early if enabled
        if self.is_maatools_platform:
            self.maatools  # Init early
        else:
            self.detect_device()

        # Connect
        self.adb_connect()
        logger.attr('AdbDevice', self.adb)

        # Package
        self.package = self.config.Emulator_PackageName
        if self.package == 'auto':
            self.detect_package()
        else:
            set_server(self.package)
        logger.attr('PackageName', self.package)
        logger.attr('Server', self.config.SERVER)

    @property
    def is_maatools_platform(self) -> bool:
        return (
                self.config.Emulator_ScreenshotMethod == 'MaaTools'
                or self.config.Emulator_ControlMethod == 'MaaTools'
        )

    def app_current(self):
        """Mock ADB app_current."""
        if self.is_maatools_platform:
            return self.package
        return ""

    def app_current_adb(self):
        """Mock ADB app_current_adb."""
        if self.is_maatools_platform:
            return self.package
        return ""

    def adb_shell(self, cmd, stream=False, recvall=True, timeout=10, rstrip=True):
        """Handle common ADB shell commands without actually using ADB."""
        if not self.is_maatools_platform:
            return ""

        if isinstance(cmd, list):
            cmd = ' '.join(cmd)

        if cmd == 'dumpsys window windows' or cmd == 'dumpsys activity top':
            # Return valid package name format
            return f'mCurrentFocus=Window{{1234 1234 {self.package}/{self.package}.MainActivity}}'
        elif cmd == 'dumpsys display':
            return "DisplayViewport{valid=true, orientation=0, deviceWidth=1280, deviceHeight=720}"
        elif cmd == 'wm size':
            return "Physical size: 1280x720"
        elif cmd == 'settings get system screen_off_timeout':
            return "2147483647"
        elif 'getprop' in cmd:
            return ""
        else:
            logger.info(f'Using MaaTools, ADB command ignored: {cmd}')
            return ""

    def screenshot_adb(self) -> np.ndarray:
        """Mock ADB screenshot method."""
        if not self.is_maatools_platform:
            return None
        return self.screenshot_maatools()

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

    @cached_property
    def maatools_builder(self):
        """Command builder instance."""
        return MaaToolsBuilder(self)

    def list_device(self):
        """Override device listing for MaaTools."""
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
            return SelectedGrids(devices)
        return super().list_device()

    def detect_device(self):
        """Verify MaaTools connection."""
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

    def get_orientation(self):
        """Get device orientation."""
        if self.is_maatools_platform:
            self._orientation = 0
            logger.attr('Device Orientation', f'0 (Normal)')
            return 0
        return super().get_orientation()

    def app_start(self):
        """Override app start."""
        if self.is_maatools_platform:
            logger.info("Using MaaTools, app start/stop not needed")
            return True
        return super().app_start()

    def app_stop(self):
        """Override app stop."""
        if self.is_maatools_platform:
            logger.info("Using MaaTools, app start/stop not needed")
            return True
        return super().app_stop()

    def app_is_running(self) -> bool:
        """Check if emulator is running."""
        if self.is_maatools_platform:
            return self.detect_platform()
        return super().app_is_running()

    def touch_init(self):
        """Initialize touch input."""
        if self.is_maatools_platform:
            logger.hr('MaaTools Touch Init')
            return True
        return super().touch_init()

    @property
    def touch_method(self):
        """Return the touch method to use."""
        if self.is_maatools_platform:
            return 'maatools'
        return super().touch_method

    def screenshot_maatools(self) -> np.ndarray:
        """Take screenshot using MaaTools."""
        if not self.is_maatools_platform:
            raise MaaToolsError("Screenshot method not set to MaaTools")

        # Request screenshot
        message = struct.pack('>H', 4) + b'SCRN'
        self.maatools.sock.sendall(message)

        # Get image size
        size_data = self.maatools.sock.recv(4)
        if len(size_data) < 4:
            raise MaaToolsError("Incomplete size data received")
        image_size = struct.unpack('>I', size_data)[0]

        # Get image data
        image_data = self._recv_exact(image_size)
        channels = image_size // (self.maatools.max_y * self.maatools.max_x)

        try:
            if channels == 4:
                # RGBA format
                image = np.frombuffer(image_data, dtype=np.uint8)
                image = image.reshape((self.maatools.max_y, self.maatools.max_x, 4))
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif channels == 3:
                # RGB format
                image = np.frombuffer(image_data, dtype=np.uint8)
                image = image.reshape((self.maatools.max_y, self.maatools.max_x, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                raise MaaToolsError(f"Unexpected image format with {channels} channels")

            image = np.ascontiguousarray(image, dtype=np.uint8)

            # **Add Vertical Flip to Match Other Platform**
            # image = cv2.flip(image, 0)

            if not hasattr(self, '_first_maatools_screenshot'):
                logger.info(
                    f'First MaaTools screenshot: '
                    f'shape={image.shape}, dtype={image.dtype}, channels={channels}'
                )
                self._first_maatools_screenshot = True
            return image
        except Exception as e:
            logger.error(f'Screenshot error: {e}')
            raise

    def _recv_exact(self, size: int) -> bytes:
        """Receive exact number of bytes."""
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
        builder.clear().down(x, y).wait(100)
        self.send_commands(builder)

        builder.clear().up(x, y)
        self.send_commands(builder)

    def click(self, button, count=1):
        """Click wrapper that uses MaaTools when enabled."""
        if self.is_maatools_platform:
            for _ in range(count):
                x, y = random_rectangle_point(button.button) if isinstance(button.button, tuple) else button.button
                self.click_maatools(x, y)
                self.sleep(0.1)
            return
        return super().click(button, count)

    def send_commands(self, builder: MaaToolsBuilder):
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

    def release_resource(self):
        """Release MaaTools resources."""
        if hasattr(self, '_maatools'):
            self._maatools.maatools_disconnect()
        del_cached_property(self, 'maatools')
        super().release_resource()

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

    def long_click(self, button, duration=1.0):
        """Long click wrapper that uses MaaTools when enabled."""
        if self.is_maatools_platform:
            x, y = random_rectangle_point(button.button) if isinstance(button.button, tuple) else button.button
            self.long_click_maatools(x, y, duration)
            return
        return super().long_click(button, duration)

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

    def swipe(self, p1, p2, duration=0.2):
        """Swipe wrapper that uses MaaTools when enabled."""
        if self.is_maatools_platform:
            self.swipe_maatools(p1, p2)
            return
        return super().swipe(p1, p2, duration)

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
        builder.send_commands(builder)

        builder.clear().move(*p2).wait(140)
        self.send_commands(builder)
        builder.clear().move(*p2).wait(140)
        self.send_commands(builder)

        builder.clear().up()
        self.send_commands(builder)

    def drag(self, p1, p2, segments=1, shake=(0, 15, 0, 0), point_random=(-10, -10, 10, 10), shake_random=(-1, -1, 1, 1),
            swipe_duration=0.25, shake_duration=0.1):
        """Drag wrapper that uses MaaTools when enabled."""
        if self.is_maatools_platform:
            self.drag_maatools(p1, p2, point_random=point_random)
            return
        return super().drag(p1, p2, segments, shake, point_random, shake_random, swipe_duration, shake_duration)

    def handle_control(self, button):
        """Handle click control."""
        if self.is_maatools_platform:
            self.click_maatools(button.x, button.y)
            return
        return super().handle_control(button)

    def get_window_size(self) -> tuple:
        """Get window size."""
        if self.is_maatools_platform:
            return self.maatools.max_x, self.maatools.max_y
        return super().get_window_size()

    @property
    def orientation(self) -> int:
        """Get current orientation from MaaTools."""
        return 0  # MaaTools assumes fixed orientation

    def adb_connect(self):
        """Connect wrapper that skips ADB for MaaTools."""
        if self.is_maatools_platform:
            logger.info("Using MaaTools, skipping ADB connection")
            return True
        return super().adb_connect()
