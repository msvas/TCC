#version 450

in vec3 TexCoords;

uniform samplerCube skybox;

layout (location = 0) out vec4 FragmentColor0;

void main()
{
	FragmentColor0 = texture(skybox, TexCoords);
}
