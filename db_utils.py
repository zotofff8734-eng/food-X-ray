from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from models import Dish, Ingredient, AsyncSessionLocal

async def get_dish_by_name(dish_name: str) -> Dish | None:
    """
    Asynchronously fetches a dish and its ingredients from the database by name.
    Performs a case-insensitive search.
    
    :param dish_name: The name of the dish to find.
    :return: A Dish object with its ingredients loaded, or None if not found.
    """
    print(f"Querying database for dish: {dish_name}")
    async with AsyncSessionLocal() as session:
        # Using .ilike() for case-insensitive matching.
        # Using joinedload to explicitly load the related ingredients in the same query.
        result = await session.execute(
            select(Dish)
            .where(Dish.name_ru.ilike(f"%{dish_name}%"))
            .options(joinedload(Dish.ingredients))
        )
        dish = result.scalars().first()
        
        if dish:
            print(f"Found dish: {dish.name_ru}")
        else:
            print("Dish not found in database.")
            
        return dish

async def add_dish(dish_data: dict) -> Dish:
    """
    Asynchronously adds a new dish and its ingredients to the database.
    This function will be used by the 'add dish' conversation handler.
    
    :param dish_data: A dictionary containing the dish details.
    :return: The newly created Dish object.
    """
    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Create a new Dish object
            new_dish = Dish(
                name_ru=dish_data.get('name'),
                calories=dish_data.get('calories'),
                recipe=dish_data.get('recipe'),
                health_impact=dish_data.get('health_impact')
            )
            
            # Placeholder for adding ingredients from a more complex dish_data
            # ingredients_list = dish_data.get('ingredients', [])
            # for ing_data in ingredients_list:
            #     ingredient = Ingredient(name_ru=ing_data.get('name'), is_healthy=ing_data.get('is_healthy'))
            #     new_dish.ingredients.append(ingredient)
            
            session.add(new_dish)
            await session.flush()
            print(f"Added new dish to DB: {new_dish.name_ru}")
            return new_dish

if __name__ == '__main__':
    import asyncio

    async def test_db_functions():
        """A simple test function to add and retrieve a dish."""
        print("--- Testing Database Utilities ---")
        
        test_dish_data = {
            'name': 'Тестовый салат',
            'calories': 150.5,
            'recipe': '1. Нарезать овощи. 2. Смешать. 3. Заправить маслом.',
            'health_impact': 'Очень полезно, содержит клетчатку и витамины.'
        }
        
        # Check if the test dish already exists
        existing_dish = await get_dish_by_name('Тестовый салат')
        
        if not existing_dish:
            print("Test dish not found, adding it now...")
            await add_dish(test_dish_data)
        else:
            print("Test dish already exists in the database.")

        # Retrieve the dish again to confirm it's there
        print("
Attempting to retrieve 'Тестовый салат'...")
        dish = await get_dish_by_name('Тестовый салат')
        
        if dish:
            print(f"
--- Retrieved Dish ---")
            print(f"  Name: {dish.name_ru}")
            print(f"  Calories: {dish.calories}")
            print(f"  Recipe: {dish.recipe}")
            print(f"  Health Impact: {dish.health_impact}")
            print("----------------------
")
        
    asyncio.run(test_db_functions())
