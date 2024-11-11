from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define the SQLite database
engine = create_engine('sqlite:///golf_data.db', echo=True)  # Set echo=True for debugging/logging
Base = declarative_base()

# Define the Player table
class Player(Base):
    __tablename__ = 'players'

    player_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    handicap = Column(Integer, nullable=False)
    average_yardages = Column(JSON, nullable=False)  # Yardages for each club as a JSON object
    club_head_speed = Column(Float, nullable=False)
    ball_speed = Column(Float, nullable=False)
    spin_rate = Column(Float, nullable=False)

# Define the HandicapGroup table
class HandicapGroup(Base):
    __tablename__ = 'handicap_groups'

    group_id = Column(Integer, primary_key=True, autoincrement=True)
    handicap_min = Column(Integer, nullable=False)
    handicap_max = Column(Integer, nullable=False)
    average_yardages = Column(JSON, nullable=False)  # Average yardages for this group
    average_club_head_speed = Column(Float, nullable=False)
    average_ball_speed = Column(Float, nullable=False)
    average_spin_rate = Column(Float, nullable=False)

# Define the InstructionalData table to store tips and advice
class InstructionalData(Base):
    __tablename__ = 'instructional_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String)  # e.g., video file name or URL
    advice = Column(Text)  # Store the extracted tips and advice from the video

# Create the tables in the database
Base.metadata.create_all(engine)

# Create a session for managing data
Session = sessionmaker(bind=engine)
session = Session()
