def get_data_query(user_id, env_id):
    return f"""
    SELECT
        pattern
    FROM
      `aixr-401704.QBRAIN.envs`
    WHERE
      id = '{env_id}' AND
      user_id = '{user_id}'
    LIMIT 1
    """
