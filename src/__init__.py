# sqlite3 override — ChromaDB requires >= 3.35; Render/Ubuntu ships 3.31
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
