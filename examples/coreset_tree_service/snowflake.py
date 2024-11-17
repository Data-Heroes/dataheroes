from datetime import datetime
import pandas as pd
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dataheroes.services.coreset_tree.dtc import CoresetTreeServiceDTC

"""
This example shows how to build a Coreset Tree on data located within a Snowflake table. 
The example uses a dataset of 2M data instances located in a table named DATA2M. The example 
shows in addition, how to write the Coreset back to a table on Snowflake.
"""


def create_connection():
    """
    Define your connectivity details here
    """
    return snowflake.connector.connect(
        user='<snowflake-user-name>',
        password='<password>',
        account='<snowflake-account>',
        warehouse='<warehouse>',
        database='<database>',
        schema='<schema>',
        role='<role-name>'
    )


def df_iterator(table_name, chunk_size):
    """
    Read data from Snowflake in chunks of chunk_size
    """
    try:
        conn = create_connection()
        offset = 0
        while True:
            query = f"SELECT * FROM {table_name} LIMIT {chunk_size} OFFSET {offset};"
            cursor = conn.cursor()
            cursor.execute(query)
            dataframe = cursor.fetch_pandas_all()
            cursor.close()
            if dataframe.empty:
                conn.close()
                break
            yield dataframe
            offset += chunk_size
    finally:
        if not conn.is_closed():
            conn.close()


def upload_df_to_snowflake(df, table_name):
    """
    Save the Coreset into a Snowflake table
    """
    try:
        conn = create_connection()
        success, n_chunks, n_rows, output = write_pandas(
            conn,
            df,
            table_name=table_name,
            auto_create_table=True
        )
        print(f"Inserted {n_rows} rows into the coreset table {table_name}.")
    finally:
        conn.close()


# main workflow
tree_chunk_size = 100_000
coreset_size = tree_chunk_size // 5

# initialize the coreset service object
service_obj = CoresetTreeServiceDTC(
    coreset_size=coreset_size,
    chunk_size=tree_chunk_size,
    optimized_for='training',
    data_params={'target': {'name': 'TARGET'}}
)

# build the tree from DataFrame iterator
service_obj.build_from_df(
    df_iterator(
        table_name='DATA2M',
        chunk_size=tree_chunk_size
    )
)

# print the tree
service_obj.print()

# get the root coreset from the Coreset tree
coreset = service_obj.get_coreset()

# create a pandas DataFrame from the coreset in order to upload it to Snowflake
coreset_df = pd.DataFrame(coreset['X'])
coreset_df.columns = [f'C{col}' for col in coreset_df.columns]
coreset_df['TARGET'] = coreset['y']
coreset_df['WEIGHTS'] = coreset['w']

# upload the coreset to Snowflake
upload_df_to_snowflake(
    df=coreset_df,
    table_name=f'CORESET_{datetime.now().strftime("%d%m%H%M")}'
)





