# src/evaluation/aspect_categories.py - Semantic clustering for product aspects
import re

CATEGORIES = {
    'battery/power': ['battery', 'power', 'charge', 'charger', 'adapter', 'cord', 'life'],
    'screen/display': ['screen', 'display', 'monitor', 'resolution', 'graphics', 'video', 'picture', 'brightness', 'retina', 'hd'],
    'keyboard/input': ['keyboard', 'keys', 'key', 'trackpad', 'touchpad', 'mouse', 'mousepad', 'button', 'buttons', 'touch', 'pad'],
    'price/value': ['price', 'prices', 'cost', 'value', 'money', 'deal'],
    'support/service': ['service', 'support', 'warranty', 'warrenty', 'customer', 'repair', 'tech', 'staff', 'shipping', 'delivery'],
    'software/OS': ['windows', 'os', 'software', 'program', 'programs', 'application', 'applications', 'apps', 'app', 'mac', 'osx',
                    'linux', 'itunes', 'iphoto', 'imovie', 'ilife', 'iwork', 'garageband', 'pages', 'office', 'internet', 'browser',
                    'vista', 'xp', 'system', 'safari', 'drivers', 'bios'],
    'performance/speed': ['performance', 'speed', 'processor', 'cpu', 'ram', 'memory', 'boot', 'boots', 'startup', 'fast', 'multitasking',
                          'processing', 'runs', 'run', 'works', 'work', 'loads', 'gaming', 'games', 'game'],
    'design/hardware': ['design', 'size', 'weight', 'build', 'built', 'case', 'quality', 'portability', 'light', 'speakers', 'sound', 'fan',
                        'port', 'ports', 'usb', 'drive', 'disk', 'dvd', 'camera', 'webcam', 'wifi', 'bluetooth', 'hardware', 'looks', 'look',
                        'appearance', 'finish', 'material', 'heat', 'cooling', 'hinge', 'motherboard', 'mother', 'board', 'mic', 'storage',
                        'space', 'colors', 'edges', 'feel', 'carry'],
    'features/usability': ['use', 'features', 'feature', 'specs', 'usability', 'navigate', 'operation', 'set'],
}


def aspect_category(term):
    """Maps free-text aspect terms into one of 9 canonical categories via keyword matching."""
    words = set(re.findall(r'[a-z0-9]+', str(term).lower()))
    for cat, keys in CATEGORIES.items():
        if words & set(keys):
            return cat
    return 'other'
