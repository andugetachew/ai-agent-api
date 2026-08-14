import sys
sys.path.insert(0, 'app')
from sqlalchemy import create_engine
from db import Base

engine = create_engine('postgresql://agent_test_user:agent_test_pass@localhost:5432/agent_db_test')
Base.metadata.create_all(bind=engine)
print('tables created')