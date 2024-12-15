import asyncio
import socket
import struct
import time
import logging
import threading
from typing import Tuple, List, Optional
from PIL import Image
import numpy as np
from functools import wraps, cached_property
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MaaToolsClient")

RETRY_TRIES = 5
DEFAULT_DELAY = 0.05  # 50ms default delay between actions

class MaaToolsError(Exception):
    """Base exception for MaaTools-related errors."""
    pass

class MaaTouchNotInstalledError(MaaToolsError):
    """Raised when MaaTouch is not installed or incompatible."""
    pass

class MaaTouchSyncTimeout(MaaToolsError):
    """Raised when a sync operation times out."""
    pass

def random_rectangle_point(bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """Generate a random point within given rectangle bounds."""
    x1, y1, x2, y2 = bounds
    return (
        np.random.randint(x1, x2),
        np.random.randint(y1, y2)
    )

def insert_swipe(p0: Tuple[int, int], p3: Tuple[int, int], speed: int = 15) -> List[Tuple[int, int]]:
    """Generate intermediate points for a smooth swipe motion."""
    x0, y0 = p0
    x3, y3 = p3

    # Calculate distance and number of steps
    distance = np.sqrt((x3 - x0) ** 2 + (y3 - y0) ** 2)
    steps = int(distance / speed)
    steps = max(steps, 2)  # Ensure at least 2 points

    points = []
    for i in range(steps):
        t = i / (steps - 1)
        x = int(x0 + (x3 - x0) * t)
        y = int(y0 + (y3 - y0) * t)
        points.append((x, y))

    return points

@dataclass
class Command:
    """Represents a single MaaTools command."""
    operation: str  # 'touch' or 'wait'
    phase: int = 0  # 0=Down, 1=Move, 3=Up
    x: int = 0
    y: int = 0
    wait_ms: int = 0  # For wait operation

    def to_bytes(self) -> bytes:
        """Convert command to MaaTools binary format with proper magic bytes and structure."""
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

def retry(func):
    """Decorator to retry operations with exponential backoff."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        for attempt in range(RETRY_TRIES):
            try:
                return func(self, *args, **kwargs)
            except (ConnectionResetError, ConnectionAbortedError) as e:
                logger.error(f"Connection error (attempt {attempt + 1}/{RETRY_TRIES}): {e}")
                if attempt < RETRY_TRIES - 1:
                    self.reconnect()
                    time.sleep(2 ** attempt)
            except MaaTouchNotInstalledError as e:
                logger.error(f"MaaTouch not installed (attempt {attempt + 1}/{RETRY_TRIES}): {e}")
                if attempt < RETRY_TRIES - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{RETRY_TRIES}): {e}")
                if attempt < RETRY_TRIES - 1:
                    time.sleep(2 ** attempt)
        raise MaaToolsError(f"Operation failed after {RETRY_TRIES} attempts")
    return wrapper

class CommandBuilder:
    """Builds sequences of touch commands."""
    DEFAULT_DELAY = 0.05

    def __init__(self, device, contact=0, handle_orientation=False):
        self.device = device
        self.contact = contact
        self.handle_orientation = handle_orientation
        self.commands: List[Command] = []
        self.delay = 0

    def clear(self):
        """Clear all pending commands."""
        self.commands = []
        self.delay = 0
        return self

    def down(self, x: int, y: int) -> 'CommandBuilder':
        """Add a touch down command."""
        self.commands.append(Command('touch', phase=0, x=x, y=y))
        return self

    def move(self, x: int, y: int) -> 'CommandBuilder':
        """Add a touch move command."""
        self.commands.append(Command('touch', phase=1, x=x, y=y))
        return self

    def up(self) -> 'CommandBuilder':
        """Add a touch up command."""
        self.commands.append(Command('touch', phase=3))  # Phase 3 = Up
        return self

    def wait(self, ms: int) -> 'CommandBuilder':
        """Add a wait command."""
        self.commands.append(Command('wait', wait_ms=ms))
        self.delay += ms
        return self

    def to_commands(self) -> List[bytes]:
        """Convert all commands to MaaTools binary format."""
        command_bytes = []
        for cmd in self.commands:
            if cmd.operation == 'wait':
                continue  # Waits are handled by sleep in send
            command_bytes.append(cmd.to_bytes())
        return command_bytes

class MaaTools:
    def __init__(self, host: str = 'localhost', port: int = 17233):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.width = 0
        self.height = 0
        self._orientation = 0
        self._orientation_lock = threading.Lock()
        self.lock = threading.Lock()

        # Protocol constants
        self.connection_magic = b'MAA\x00'
        self.screencap_magic = b'SCRN'
        self.size_magic = b'SIZE'
        self.terminate_magic = b'TERM'
        self.toucher_magic = b'TUCH'
        self.version_magic = b'VERN'

        # Screenshot related
        self._screenshot_lock = threading.Lock()
        self._last_screenshot: Optional[Image.Image] = None

    @cached_property
    def command_builder(self) -> CommandBuilder:
        """Get a command builder instance."""
        return CommandBuilder(self)

    @retry
    def connect(self):
        """Establish connection and perform handshake."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to MaaTools server at {self.host}:{self.port}")
            self._handshake()
            self.width, self.height = self._get_window_size()
            logger.info(f"Window size: {self.width}x{self.height}")
        except Exception as e:
            self.sock = None
            raise MaaToolsError(f"Connection failed: {e}")

    def reconnect(self):
        """Reconnect to the server."""
        self.disconnect()
        self.connect()

    def disconnect(self):
        """Close the connection."""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
            finally:
                self.sock = None

    def send_commands(self, builder: CommandBuilder):
        """Send commands as MaaTools binary format."""
        command_bytes = builder.to_commands()

        with self.lock:
            for cmd_bytes in command_bytes:
                self.sock.sendall(cmd_bytes)
                # Handle wait commands through actual sleep
                if builder.delay > 0:
                    time.sleep(builder.delay / 1000)

            # Always add a small delay between command sequences
            time.sleep(builder.DEFAULT_DELAY)

    async def send_commands_async(self, builder: CommandBuilder):
        """Send commands asynchronously."""
        command_bytes = builder.to_commands()

        # Create event loop for async operations
        loop = asyncio.get_event_loop()

        async def send_command(cmd_bytes: bytes):
            # Use loop.run_in_executor for socket operations since they're blocking
            await loop.run_in_executor(None, lambda: self.sock.sendall(cmd_bytes))

        async with asyncio.Lock():  # Async lock for thread safety
            for cmd_bytes in command_bytes:
                await send_command(cmd_bytes)
                if builder.delay > 0:
                    await asyncio.sleep(builder.delay / 1000)

            # Add default delay
            await asyncio.sleep(builder.DEFAULT_DELAY)

    @retry
    def click_maatools(self, x: int, y: int):
        """Perform a click operation."""
        builder = self.command_builder
        builder.clear().down(x, y).wait(10)
        self.send_commands(builder)

        builder.clear().up()
        self.send_commands(builder)

    @retry
    def long_click_maatools(self, x: int, y: int, duration: float = 1.0):
        """Perform a long click operation."""
        duration_ms = int(duration * 1000)
        builder = self.command_builder
        builder.clear().down(x, y).wait(duration_ms)
        self.send_commands(builder)

        builder.clear().up()
        self.send_commands(builder)

    @retry
    def swipe_maatools(self, p1: Tuple[int, int], p2: Tuple[int, int], duration: float = 1.0):
        """Perform a swipe operation."""
        points = insert_swipe(p0=p1, p3=p2)
        builder = self.command_builder

        builder.clear().down(*points[0]).wait(10)
        self.send_commands(builder)

        for point in points[1:]:
            builder.clear().move(*point).wait(10)
            self.send_commands(builder)

        builder.clear().up()
        self.send_commands(builder)

    @retry
    def drag_maatools(self, p1: Tuple[int, int], p2: Tuple[int, int],
                      point_random: Tuple[int, int, int, int] = (-10, -10, 10, 10)):
        """Perform a drag operation with randomization."""
        p1 = np.array(p1) - random_rectangle_point(point_random)
        p2 = np.array(p2) - random_rectangle_point(point_random)
        points = insert_swipe(tuple(p1), tuple(p2), speed=20)

        builder = self.command_builder

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

    async def click_maatools_async(self, x: int, y: int):
        """Perform a click operation asynchronously."""
        builder = self.command_builder
        builder.clear().down(x, y).wait(10)
        await self.send_commands_async(builder)

        builder.clear().up()
        await self.send_commands_async(builder)

    async def swipe_maatools_async(self, p1: Tuple[int, int], p2: Tuple[int, int], duration: float = 1.0):
        """Perform a swipe operation asynchronously."""
        points = insert_swipe(p0=p1, p3=p2)
        builder = self.command_builder

        builder.clear().down(*points[0]).wait(10)
        await self.send_commands_async(builder)

        for point in points[1:]:
            builder.clear().move(*point).wait(10)
            await self.send_commands_async(builder)

        builder.clear().up()
        await self.send_commands_async(builder)

    async def drag_maatools_async(self, p1: Tuple[int, int], p2: Tuple[int, int],
                                  point_random: Tuple[int, int, int, int] = (-10, -10, 10, 10)):
        """Perform a drag operation asynchronously."""
        p1 = np.array(p1) - random_rectangle_point(point_random)
        p2 = np.array(p2) - random_rectangle_point(point_random)
        points = insert_swipe(tuple(p1), tuple(p2), speed=20)

        builder = self.command_builder

        builder.clear().down(*points[0]).wait(10)
        await self.send_commands_async(builder)

        for point in points[1:]:
            builder.clear().move(*point).wait(10)
            await self.send_commands_async(builder)

        builder.clear().move(*p2).wait(140)
        await self.send_commands_async(builder)
        builder.clear().move(*p2).wait(140)
        await self.send_commands_async(builder)

        builder.clear().up()
        await self.send_commands_async(builder)

    @retry
    def screenshot(self, save_path: str = 'screenshot.raw') -> bytes:
        """Capture a screenshot and return raw bytes."""
        try:
            with self._screenshot_lock:
                # Construct and send SCRN command
                payload_length = 4
                message = struct.pack('>H', payload_length) + self.screencap_magic

                with self.lock:
                    self.sock.sendall(message)

                    # Get image size
                    size_data = self._recv_exact(4)
                    image_size = struct.unpack('>I', size_data)[0]

                    # Get image data
                    image_data = self._recv_exact(image_size)

                    # Save if path provided
                    if save_path:
                        with open(save_path, 'wb') as f:
                            f.write(image_data)

                    return image_data
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            raise MaaToolsError(f"Screenshot error: {e}")

    @retry
    def screenshot_pillow(self, save_path: str = 'screenshot.png') -> Optional[Image.Image]:
        """Capture a screenshot and return as PIL Image."""
        try:
            image_data = self.screenshot(save_path='screenshot.raw')
            # Calculate expected sizes
            expected_size_rgb = self.width * self.height * 3
            expected_size_rgba = self.width * self.height * 4

            if len(image_data) == expected_size_rgb:
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image_array = image_array.reshape((self.height, self.width, 3))
                mode = 'RGB'
            elif len(image_data) == expected_size_rgba:
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image_array = image_array.reshape((self.height, self.width, 4))
                mode = 'RGBA'
            else:
                raise ValueError(f"Unexpected image data size: {len(image_data)}")

            image = Image.fromarray(image_array, mode)
            if save_path:
                image.save(save_path)

            self._last_screenshot = image
            return image

        except Exception as e:
            logger.error(f"Screenshot conversion failed: {e}")
            return None

    async def screenshot_async(self, save_path: str = 'screenshot.png') -> Optional[Image.Image]:
        """Capture a screenshot asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.screenshot_pillow(save_path))

    def _handshake(self):
        """Perform protocol handshake."""
        with self.lock:
            self.sock.sendall(self.connection_magic)
            response = self._recv_exact(4)
            if response != b'OKAY':
                raise MaaToolsError(f"Handshake failed: {response}")

    def _get_window_size(self) -> Tuple[int, int]:
        """Get the window dimensions."""
        with self.lock:
            message = struct.pack('>H', 4) + self.size_magic
            self.sock.sendall(message)
            size_data = self._recv_exact(4)
            return struct.unpack('>HH', size_data)

    def _recv_exact(self, num_bytes: int) -> bytes:
        """Receive exact number of bytes."""
        data = b''
        while len(data) < num_bytes:
            packet = self.sock.recv(num_bytes - len(data))
            if not packet:
                break
            data += packet
        return data

    @property
    def orientation(self) -> int:
        """Get the current orientation."""
        return self._orientation

    @orientation.setter
    def orientation(self, value: int):
        """Set the orientation and trigger necessary updates."""
        with self._orientation_lock:
            if self._orientation != value:
                logger.info(f"Orientation changed: {self._orientation} -> {value}")
                self._orientation = value
                self._on_orientation_change()

    def _on_orientation_change(self):
        """Handle orientation changes."""
        # Re-initialize screen dimensions
        self.width, self.height = self._get_window_size()
        logger.info(f"Updated dimensions after orientation change: {self.width}x{self.height}")

    def reset(self):
        """Reset the touch state."""
        builder = self.command_builder
        builder.clear()
        builder.reset().commit()
        self.send_commands(builder)

    def terminate(self):
        """Send termination command to the server."""
        try:
            payload_length = 4  # 4 bytes for magic
            message = struct.pack('>H', payload_length) + self.terminate_magic

            with self.lock:
                self.sock.sendall(message)
            logger.info("Sent termination command to server")
        except Exception as e:
            logger.error(f"Failed to send termination command: {e}")
            raise MaaToolsError(f"Termination error: {e}")