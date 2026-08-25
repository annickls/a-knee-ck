"""A headlight shader for the bone meshes.

pyqtgraph's built-in ``shaded`` program hardcodes its light direction to
``vec3(1, -1, -1)`` in eye space -- behind and below the camera -- and ignores
``GLViewWidget.opts['lightPosition']``, ``ambient``, ``diffuse`` and friends
entirely. Those options exist on the view but nothing consumes them, which is
why the original's lighting block had no effect and the bones read almost black.

This program puts the light just off the camera axis and evaluates a two-sided
Lambert term. Two-sided matters: these meshes come from CT segmentation and
their triangle winding is not reliably outward, so a one-sided term leaves
patches of the surface unlit.
"""

from __future__ import annotations

from pyqtgraph.opengl.shaders import FragmentShader, ShaderProgram, VertexShader

BONE_SHADER = "kneeBone"

_VERTEX = """
    uniform mat4 u_mvp;
    uniform mat3 u_normal;
    attribute vec4 a_position;
    attribute vec3 a_normal;
    attribute vec4 a_color;
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        v_normal = normalize(u_normal * a_normal);
        v_color = a_color;
        gl_Position = u_mvp * a_position;
    }
"""

_FRAGMENT = """
    #ifdef GL_ES
    precision mediump float;
    #endif
    varying vec4 v_color;
    varying vec3 v_normal;
    void main() {
        vec3 n = normalize(v_normal);
        vec3 light = normalize(vec3(-0.30, 0.40, 1.0));
        float diffuse = abs(dot(n, light));
        float rim = pow(1.0 - abs(n.z), 3.0) * 0.16;
        float shade = 0.42 + 0.56 * diffuse + rim;
        gl_FragColor = vec4(clamp(v_color.rgb * shade, 0.0, 1.0), v_color.a);
    }
"""


def register() -> str:
    """Register the bone shader with pyqtgraph and return its name.

    ``ShaderProgram`` self-registers by name on construction, so this is
    idempotent and safe to call from every scene that needs it.
    """
    if BONE_SHADER not in ShaderProgram.names:
        ShaderProgram(BONE_SHADER, [VertexShader(_VERTEX), FragmentShader(_FRAGMENT)])
    return BONE_SHADER
