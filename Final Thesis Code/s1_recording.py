"""
Air Writing Data Collection and Preprocessing System
Combines real-time data collection from flex sensors and button input
with preprocessing pipeline for machine learning

Hardware layout — 5 ADS1115 chips across 2 I2C buses
------------------------------------------------------
I2C Bus 1 (Pi pin 3=SDA, pin 5=SCL):
    Chip #1  Flex sensor 1   ADDR→GND   0x48
    Chip #2  Flex sensor 2   ADDR→VDD   0x49

I2C Bus 2 (Pi pin 15=SDA GPIO22, pin 16=SCL GPIO23):
    Chip #3  Flex sensor 3   ADDR→GND   0x48
    Chip #4  Flex sensor 4   ADDR→VDD   0x49
    Chip #5  Button ladder   ADDR→SDA   0x4A

/boot/config.txt additions required:
    dtparam=i2c_arm_baudrate=400000
    dtoverlay=i2c-gpio,bus=2,i2c_gpio_sda=22,i2c_gpio_scl=23
"""

import board
import busio
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.ads1x15 import Mode
from adafruit_ads1x15.analog_in import AnalogIn
import time
from datetime import datetime
from typing import Tuple, Dict, List


# =============================================================================
# CONSTANTS
# =============================================================================

DIRECTIONS = {
    '1': 'N',
    '2': 'NE',
    '3': 'E',
    '4': 'SE',
    '5': 'S',
    '6': 'SW',
    '7': 'W',
    '8': 'NW',
}

START_POSITIONS = {
    '1': 'N',
    '2': 'NE',
    '3': 'E',
    '4': 'SE',
    '5': 'S',
    '6': 'SW',
    '7': 'W',
    '8': 'NW',
    '9': 'center',
}


# =============================================================================
# INPUT HELPERS
# =============================================================================

def prompt_direction() -> str:
    """Prompt user to select one of 8 cardinal directions."""
    print("\nSelect gesture direction:")
    for key, label in DIRECTIONS.items():
        print(f"  {key} = {label}")
    while True:
        choice = input("Enter number (1-8): ").strip()
        if choice in DIRECTIONS:
            return DIRECTIONS[choice]
        print("  Invalid input. Please enter a number between 1 and 8.")


def prompt_start_position() -> str:
    """Prompt user to select a start position (8 cardinal + center)."""
    print("\nSelect start position:")
    for key, label in START_POSITIONS.items():
        print(f"  {key} = {label}")
    while True:
        choice = input("Enter number (1-9): ").strip()
        if choice in START_POSITIONS:
            return START_POSITIONS[choice]
        print("  Invalid input. Please enter a number between 1 and 9.")


# =============================================================================
# DATA COLLECTION
# =============================================================================

class AirWritingCollector:
    """Collects air writing data from flex sensors and button inputs.

    Bus 1 (hardware I2C): sensors 1 and 2
    Bus 2 (software I2C on GPIO 22/23): sensors 3, 4 and button
    Each chip is one channel only — no multiplexing, runs continuous mode.
    """

    def __init__(self):
        # --- I2C Bus 1 — hardware I2C on pins 3 (SDA) and 5 (SCL) ---
        self.i2c1 = busio.I2C(board.SCL, board.SDA)

        # Chip #1 — flex sensor 1 — ADDR → GND → 0x48, buttons
        self.ads1 = ADS.ADS1115(self.i2c1, address=0x48, data_rate=860)
        self.ads1.mode = Mode.CONTINUOUS
        self.button_input = AnalogIn(self.ads1, 0)

        # Chip #2 — flex sensor 2 — ADDR → VDD → 0x49, index finger
        self.ads2 = ADS.ADS1115(self.i2c1, address=0x49, data_rate=860)
        self.ads2.mode = Mode.CONTINUOUS
        self.sensor2 = AnalogIn(self.ads2, 0)

        # Chip #3 — button resistor ladder — ADDR → SDA → 0x4A, lateral wrist
        self.ads5 = ADS.ADS1115(self.i2c1, address=0x4A, data_rate=860)
        self.ads5.mode = Mode.CONTINUOUS
        self.sensor5 = AnalogIn(self.ads5, 0)

        # --- I2C Bus 2 — software I2C on GPIO 22 (SDA) and GPIO 23 (SCL) ---
        # busio.I2C does not support software I2C GPIO pins — open /dev/i2c-2 directly
        # Requires dtoverlay=i2c-gpio,bus=2,i2c_gpio_sda=22,i2c_gpio_scl=23
        # in /boot/config.txt and a reboot. Verify with: sudo i2cdetect -y 2
        from adafruit_extended_bus import ExtendedI2C
        self.i2c2 = ExtendedI2C(2)   # opens /dev/i2c-2

        # Chip #4 — flex sensor 3 — ADDR → GND → 0x48, stretch
        self.ads3 = ADS.ADS1115(self.i2c2, address=0x48, data_rate=860)
        self.ads3.mode = Mode.CONTINUOUS
        self.sensor3 = AnalogIn(self.ads3, 0)

        # Chip #5 — flex sensor 4 — ADDR → VDD → 0x49, wrist top
        self.ads4 = ADS.ADS1115(self.i2c2, address=0x49, data_rate=860)
        self.ads4.mode = Mode.CONTINUOUS
        self.sensor4 = AnalogIn(self.ads4, 0)

        

        # Button voltage thresholds
        # A = No button pressed (baseline/rest)
        # B = Slow speed, Long length
        # C = Fast speed, Long length
        # D = Slow speed, Short length
        # E = Fast speed, Short length
        self.BUTTON_THRESHOLDS = {
            'A': (24000, 29000),
            'B': (0,  5000),
            'C': (11000,  15000),
            'D': (16600,  18400),
            'E': (19000, 20000),
        }
    
    def detect_button(self, voltage_value):
        """Detect which button is pressed based on voltage reading"""
        for label, (low, high) in self.BUTTON_THRESHOLDS.items():
            if low <= voltage_value < high:
                return label
        return ''
    
    def collect_data(self, direction=None, start_position=None, sample_rate=250):
        """
        Collect air writing data and save to file.

        Parameters
        ----------
        direction : str, optional
            One of 8 cardinal directions (N, NE, E, SE, S, SW, W, NW).
            If not provided, user will be prompted to select.
        start_position : str, optional
            One of 8 cardinal directions or 'center'.
            If not provided, user will be prompted to select.
        sample_rate : int
            Sampling rate in Hz (default 250).
        """
        if direction is None:
            direction = prompt_direction()
        if start_position is None:
            start_position = prompt_start_position()

        # Generate filename with timestamp (YYYYMMDD_HHMM format)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f'{direction}_{start_position}_{timestamp}.txt'
        
        print("\n" + "=" * 60)
        print("Air Writing Data Collection - Resistor Ladder System")
        print("=" * 60)
        print("\nButton Mappings:")
        print("  A = No button pressed (still / baseline)")
        print("  B = Slow speed, Long length")
        print("  C = Fast speed, Long length")
        print("  D = Slow speed, Short length")
        print("  E = Fast speed, Short length")
        print(f"\nDirection:      {direction}")
        print(f"Start position: {start_position}")
        print(f"Recording to:   {filename}")
        print(f"Sampling at {sample_rate} Hz...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        sample_count = 0
        start_time = time.time()
        prev_marker = 'A'
        
        try:
            with open(filename, 'w') as file:
                # Write header
                file.write(f"% Air Writing Data Collection\n")
                file.write(f"% Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                file.write(f"% Direction: {direction}\n")
                file.write(f"% Start Position: {start_position}\n")
                file.write(f"% Sample Rate: {sample_rate} Hz\n")
                file.write(f"% Channels: 4 flex sensors\n")
                file.write(f"% Button System: Resistor Ladder (5 states: A-E)\n")
                file.write(f"% Button Labels: A=Still/None, B=Slow/Long, C=Fast/Long, D=Slow/Short, E=Fast/Short\n")
                file.write("Timestamp,Channel1,Channel2,Channel3,Channel4,Marker\n")
                
                target_interval = 1.0 / sample_rate
                next_sample_time = start_time
                buffer = []
                FLUSH_EVERY = 50   # write to disk every 50 samples

                while True:
                    # read all 5 channels as fast as possible
                    t        = time.time() - start_time
                    s1       = self.sensor5.value #lateral
                    s2       = self.sensor2.value #index
                    s3       = self.sensor3.value #stretch
                    s4       = self.sensor4.value #wrist
                    btn      = self.button_input.value
                    marker   = self.detect_button(btn)

                    buffer.append(f"{t:.6f},{s1},{s2},{s3},{s4},{marker}\n")

                    # flush buffer to disk every FLUSH_EVERY samples
                    if len(buffer) >= FLUSH_EVERY:
                        file.write(''.join(buffer))
                        file.flush()
                        buffer.clear()

                    # heartbeat — only print once per second
                    if sample_count % sample_rate == 0 and sample_count > 0:
                        print(f"Time: {t:.1f}s | Marker: {marker} | Rate: {sample_count/t:.0f} Hz")

                    # non-A marker alert (throttled — only on transition)
                    if marker and marker != 'A' and (sample_count == 0 or prev_marker == 'A'):
                        print(f">>> MARKER: {marker} <<<")
                    prev_marker = marker

                    sample_count += 1

                    # pace to target sample rate
                    next_sample_time += target_interval
                    sleep_time = next_sample_time - time.time()
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
        except KeyboardInterrupt:
            # flush any remaining buffered samples
            if buffer:
                file.write(''.join(buffer))
            print("\n\nStopped by user")
        finally:
            end_time = time.time()
            duration = end_time - start_time
            print("\n" + "=" * 60)
            print(f"Recording complete!")
            print(f"Direction:      {direction}")
            print(f"Start position: {start_position}")
            print(f"Duration:       {duration:.2f} seconds")
            print(f"Total samples:  {sample_count}")
            print(f"Actual rate:    {sample_count/duration:.2f} Hz")
            print(f"Saved to:       {filename}")
            print("=" * 60)
            
            return filename


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    collector = AirWritingCollector()
    collector.collect_data()
