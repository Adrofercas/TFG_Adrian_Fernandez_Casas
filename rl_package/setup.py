from setuptools import find_packages, setup

package_name = 'rl_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'gymnasium', 'stable-baselines3'],
    zip_safe=True,
    maintainer='afernandez',
    maintainer_email='afernandez@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'one_joint = rl_package.one_joint:main',
            'entrenamiento = rl_package.entrenamiento:main',
            'record_move = rl_package.record_move:main',
            'perfil_trapezoidal = rl_package.perfil_trapezoidal:main'
        ],
    },
)
