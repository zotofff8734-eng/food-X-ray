from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    REAL,
    BOOLEAN,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os

# Get the database URL from environment variable, default to local SQLite file
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///my_food_database.db")

# --- Async Engine Setup ---
# The async engine is what our application will use to interact with the DB
async_engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)

# --- Base Class for Declarative Models ---
# All our ORM models will inherit from this class
Base = declarative_base()


# --- ORM Model for the 'dishes' table ---
class Dish(Base):
    __tablename__ = "dishes"

    id = Column(Integer, primary_key=True)
    name_ru = Column(String, nullable=False, unique=True)
    calories = Column(REAL)
    health_impact = Column(String)
    recipe = Column(String)

    # This creates a one-to-many relationship: one Dish has many Ingredients
    ingredients = relationship("Ingredient", back_populates="dish", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Dish(id={self.id}, name='{self.name_ru}')>"


# --- ORM Model for the 'ingredients' table ---
class Ingredient(Base):
    __tablename__ = "ingredients"

    id = Column(Integer, primary_key=True)
    dish_id = Column(Integer, ForeignKey("dishes.id"), nullable=False)
    name_ru = Column(String, nullable=False)
    is_healthy = Column(BOOLEAN)
    reasoning = Column(String)

    # This creates a many-to-one relationship back to the Dish
    dish = relationship("Dish", back_populates="ingredients")

    def __repr__(self):
        return f"<Ingredient(id={self.id}, name='{self.name_ru}', dish_id={self.dish_id})>"


# --- Helper function to create all tables ---
# This can be used if the main database.py script is not used.
async def create_all_tables():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("All tables created successfully.")

if __name__ == "__main__":
    # Example of how to run the table creation
    import asyncio

    # This will create the tables if they don't exist, based on the models defined above.
    # It's an alternative to the pure SQL script in database.py.
    print("Running async table creation...")
    asyncio.run(create_all_tables())
    print("Done.")
