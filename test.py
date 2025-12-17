import jax.numpy as jnp

def get_demo_data():
    return [
        # Module 0
        [
            # Equation 0
            [
                lambda x: x*x,  # eqattrs["callable"]
                [
                    [0, 1, 0],  # [module_index, param_index, field_index]
                    [2, 0, 3]
                ],
                [0, 2, 1],  # return_index_map
                [None, 0],  # axis_def
                0  # method_id
            ],

            # Equation 1
            [
                lambda x: x*2,
                [
                    [1, 0, 0]
                ],
                [0, 1, 0],
                [None],
                1
            ]
        ],

        # Module 1
        [
            [
                lambda x: x*3,
                [
                    [1, 2, 0]
                ],
                [1, 0, 0],
                [0],
                0
            ]
        ]
    ]



def get_demo_db():
    return [
        [#mod
            [#f
                [#val
                    0,0,0,jnp.array([0,0,0])
                ],
                [#axis
                    None,None,None,0
                ]
            ]
        ],
        [  # mod
            [  # f
                [  # val
                    0, 0, 0, jnp.array([0, 0, 0])
                ],
                [  # axis
                    None, None, None, 0
                ]
            ],
            [  # f
                [  # val
                    0, 0, 0, jnp.array([0, 0, 0])
                ],
                [  # axis
                    None, None, None, 0
                ]
            ]
        ],
    ]