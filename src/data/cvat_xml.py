import torch
import xml.etree.ElementTree as ET
from typing import List, Optional
from numpy.typing import ArrayLike


class Track:
    """ Abstract skeleton track representation in CVAT XML format.
        This class is not meant to be instantiated. Use its object class-specific subclasses instead (PersonTrack, BallTrack, TableTrack).

        Example of parsing object tracks from a CVAT XML using ElementTree and XPath:

            xml = ET.parse("data.xml")
            player1 = [PersonTrack.load(track)
                       for track in xml.findall("track[@label='Person']/skeleton/attribute[@name='Role'][.='Player 1']/../..")]
            player2 = [PersonTrack.load(track)
                       for track in xml.findall("track[@label='Person']/skeleton/attribute[@name='Role'][.='Player 2']/../..")]
            balls = [BallTrack.load(track)
                     for track in xml.findall("track[@label='Ball']/skeleton/attribute[@name='Main'][.='true']/../..")]
            table = [TableTrack.load(track)
                     for track in xml.findall("track[@label='Table']")]
    """

    class Point:
        """ 2D point of a CVAT skeleton with its attributes
        """
        def __init__(self, x: float, y: float, *, is_keyframe = False, is_occluded = False, is_outside = False):
            self.xy = torch.tensor((x, y), dtype=torch.float32)
            self.is_keyframe = is_keyframe
            self.is_occluded = is_occluded
            self.is_outside = is_outside

        @staticmethod
        def parse(xml: ET.Element):
            """ Parses an XML representation of a Point.
            """
            x, y = map(float, xml.attrib["points"].split(","))
            return Track.Point(x, y,
                               is_keyframe=xml.attrib["keyframe"] == "1",
                               is_occluded=xml.attrib["occluded"] == "1",
                               is_outside=xml.attrib["outside"] == "1")

        @property
        def x(self):
            return self.xy[0].item()

        @property
        def y(self):
            return self.xy[1].item()

        def numpy(self) -> Optional[ArrayLike]:
            """ Returns a numpy ndarray with point coordinates or None if the point is marked as "outside".
            """
            if self.is_outside:
                return None
            return self.xy.to(int).numpy()
        
        def __repr__(self):
            return f"x : {self.x} {self.y} "


    class Skeleton:
        """ CVAT skeleton representation.
        """
        def __init__(self, points, is_keyframe=False):
            self.points : List[Track.Point] = points
            self.is_keyframe = is_keyframe

        def __len__(self):
            """ Returns the number of points in the skeleton.
            """
            return len(self.points)

        def __getitem__(self, i):
            return self.points[i]

        def parse_attributes(self, xml_element: ET.Element):
            pass

        def is_outside(self) -> bool:
            return all(pt is None or pt.is_outside for pt in self.points)

    @staticmethod
    def get_attribute(xml: ET.Element, name: str) -> Optional[str]:
        """ Returns an attribute value by its name scanning all <attribute> tags.
        """
        values = { attr.text
                   for attr in xml.findall(f"./skeleton/attribute[@name='{name}']") }
        if not values:
            return None
        if len(values) > 1:
            raise RuntimeError(f"Got multiple values for '{name}' attribute: {values}")
        return values.pop()

    @classmethod
    def get_points_labels(cls) -> List[str]:
        raise NotImplementedError()

    @classmethod
    def parse_attributes(cls, xml_element: ET.Element, instance):
        pass

    @classmethod
    def load(cls, xml_element: ET.Element):
        """ Loads the track from an XML element.
        """
        track = cls(xml_element.attrib["label"], int(xml_element.attrib["id"]))
        track.data = [None] * 256
        max_frame = 0

        pts_order = {
            label: i
            for i, label in enumerate(cls.get_points_labels())
        }

        # parse skeletons
        boxes = xml_element.findall("box")
        if boxes:
            # parsing single-point skeletons for balls from automatic CVAT tracker (not used anymore)
            for box in xml_element.findall("box"):
                assert len(track.pts_labels) == 1
                frame = int(box.attrib["frame"])
                is_outside = box.attrib["outside"] == "1"
                if is_outside:
                    track.set(frame, None)
                else:
                    x = 0.5 * (float(box.attrib["xbr"]) + float(box.attrib["xtl"]))
                    y = 0.5 * (float(box.attrib["ybr"]) + float(box.attrib["ytl"]))
                    skeleton = cls.Skeleton([
                        Track.Point(x, y,
                                    is_keyframe=True,
                                    is_occluded=box.attrib["occluded"] == "1")
                    ], True)
                    track.set(frame, skeleton)
                max_frame = max(max_frame, frame)
        else:
            skeletons = xml_element.findall("skeleton")
            assert skeletons, "Neither <box> nor <skeleton> tags found"
            for skeleton_xml in skeletons:
                frame = int(skeleton_xml.attrib["frame"])

                # reorder points to follow the order of labels
                unordered_pts = skeleton_xml.findall("points")
                points = [Track.Point(0, 0, is_outside=True)] * len(pts_order)
                for pt_xml in unordered_pts:
                    points[pts_order[pt_xml.attrib["label"]]] = Track.Point.parse(pt_xml)

                skeleton = cls.Skeleton(points, skeleton_xml.attrib["keyframe"] == "1")
                if skeleton.is_outside():
                    track.set(frame, None)
                else:
                    track.set(frame, skeleton)

                # parse mutable (per-frame) attributes
                skeleton.parse_attributes(skeleton_xml)

                max_frame = max(max_frame, frame)

        # remove unused entries
        track.data = track.data[:(max_frame - track.start_frame + 1)]

        try:
            cls.parse_attributes(xml_element, track)
        except:
            raise ValueError(f"Could not parse attributes of {track} track")
        return track

    def __init__(self, label: str, id: int):
        self.label = label
        self.id = id
        self.start_frame = None
        self.data : List[Track.Skeleton] = []
        self.pts_labels = self.get_points_labels()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> Optional[Skeleton]:
        return self.data[i]

    def __call__(self, frame_number: int) -> Optional[Skeleton]:
        """ Returns the track value using absolute frame number
        """
        return self.data[frame_number - self.start_frame]

    def last_frame(self, allow_outside=True) -> int:
        """ Returns the last frame index.
        """
        if allow_outside:
            return self.start_frame + len(self.data) - 1

        n = len(self.data) - 1
        while n > 0 and self.data[n] is None:
            n -= 1
        return self.start_frame + n

    def set(self, frame: int, skeleton: Optional[Skeleton]):
        """ Changes track at a given frame.
        """
        if self.start_frame is None:
            self.start_frame = frame
        elif frame < self.start_frame:
            raise RuntimeError("Frame indices are out of order")

        # make room for new entries
        i = frame - self.start_frame
        if i >= len(self.data):
            self.data += [None] * (i - len(self.data) + 1)

        self.data[i] = skeleton

    def __repr__(self):
        return f"{self.label} #{self.id} [{self.start_frame}-{self.last_frame()}]"


class BallTrack(Track):
    class Skeleton(Track.Skeleton):
        def parse_attributes(self, xml_element: ET.Element):
            super().parse_attributes(xml_element)
            node = xml_element.find("./attribute[@name='Diameter']")
            if node is None:
                self.diameter = None
            else:
                self.diameter = float(node.text)
                if self.diameter == 0:
                    self.diameter = None

    @classmethod
    def get_points_labels(cls) -> List[str]:
        return ["1"]

    @classmethod
    def parse_attributes(cls, xml_element: ET.Element, instance):
        instance.is_main = cls.get_attribute(xml_element, "Main").lower() == "true"

    def __init__(self, label: str, id: int):
        if label != "Ball":
            raise ValueError(f"Unsupported label: {self.label}")
        self.is_main = False
        super().__init__(label, id)


class PersonTrack(Track):
    @classmethod
    def get_points_labels(cls) -> List[str]:
        return ["nose", "L shoulder", "R shoulder", "L elbow", "R elbow", "L wrist", "R wrist", "L hip", "R hip", "L knee", "R knee", "L ankle", "R ankle"]

    @classmethod
    def parse_attributes(cls, xml_element: ET.Element, instance):
        instance.role = cls.get_attribute(xml_element, "Role")

    def __init__(self, label: str, id: int):
        if label != "Person":
            raise ValueError(f"Unsupported label: {self.label}")
        self.role = "Other"
        super().__init__(label, id)

    def __repr__(self):
        return f"{self.role} #{self.id} [{self.start_frame}-{self.last_frame()}]"


class TableTrack(Track):
    @classmethod
    def get_points_labels(cls) -> List[str]:
        return ["1", "2", "3", "4"]


class EventSequence:
    """ Temporal sequence of tags (events)
    """

    PLAYER_1 = 1 << 0
    PLAYER_2 = 1 << 1
    SERVE = 1 << 2
    LET_SERVE = 1 << 3
    VOID_SERVE = 1 << 4
    BALL_PASS = 1 << 5
    POINT = 1 << 6
    MISTAKE = 1 << 7
    FOREHAND = 1 << 8
    BACKHAND = 1 << 9

    ALL_EVENTS = {
        "player 1": PLAYER_1,
        "player 2": PLAYER_2,
        "serve": SERVE,
        "let serve": LET_SERVE,
        "void serve": VOID_SERVE,
        "ball pass": BALL_PASS,
        "point": POINT,
        "mistake": MISTAKE,
        "forehand": FOREHAND,
        "backhand": BACKHAND,
    }

    def __init__(self, xml: Optional[ET.ElementTree] = None):
        self.events = {}
        if xml is not None:
            self.add(xml)

    def __getitem__(self, i: int) -> List[str]:
        """ Returns a list of tag names at a given frame in the sequence.
        """
        events = self.events.get(i)
        if events is None:
            return []
        return [ name
                 for name, flag in EventSequence.ALL_EVENTS.items()
                 if flag & events ]

    def add(self, xml: ET.ElementTree):
        """ Adds an XML file contents to the current sequence.
        """
        last_player_evt_id = None
        for image in xml.getroot().findall("image/[tag]"):
            id = int(image.attrib["id"])

            # get the bitmask corresponding to the current set of events
            try:
                mask = sum(EventSequence.ALL_EVENTS[tag.attrib["label"]]
                           for tag in image.findall("tag"))
            except KeyError:
                raise ValueError(f"Invalid tag for image #{id}")

            # store it at the current frame or at a frame before
            if mask & (EventSequence.PLAYER_1 + EventSequence.PLAYER_2):
                self.events[id] = mask
                last_player_evt_id = id
            else:
                self.events[last_player_evt_id] = self.events[last_player_evt_id] | mask
