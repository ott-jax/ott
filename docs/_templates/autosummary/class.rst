{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

    {% block methods %}
    {% if methods %}
    .. rubric:: Methods
    .. autosummary::
       :toctree: .
    {% for item in methods %}
        {% if (item != '__init__') and (item != 'tree_flatten') and (item != 'tree_unflatten') %}
            ~{{ fullname }}.{{ item }}
        {% endif %}
    {% endfor %}
    {% endif %}
    {% endblock %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
       :toctree: .
    {% for item in attributes %}
       ~{{ fullname }}.{{ item }}
    {% endfor %}
    {% endif %}
    {% endblock %}
