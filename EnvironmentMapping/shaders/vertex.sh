#version 450

layout (location = 0) in vec3 VertexPosition;

uniform mat4 Model;
uniform mat4 View;
uniform mat4 Projection;

out vec3 TexCoords;

void main()
{
	mat4 MVP = Projection * Model;
	gl_Position = MVP * vec4(VertexPosition, 1.0);

    TexCoords = VertexPosition;
}
